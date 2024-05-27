# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import os
import sys
import pprint

import numpy as np
import torch

from common_utils.hparams import set_hparams, hparams
from common_utils.checkpoint import build_object_from_config
from act_group import ActGroup
from create import create_envs, create_threads
from eval import evaluate
import common_utils
import rela
import r2d2
import utils

torch.backends.cudnn.benchmark = True

set_hparams()
if hparams['off_belief']:
    hparams['vdn'] = False
    hparams['multi_step'] = 1
    hparams['net']['cls'] = "net.LSTMNet"
    hparams['net']['public'] = True
    hparams['shuffle_color'] = False
if hparams['vdn']:
    hparams['batchsize'] = int(np.round(hparams['batchsize'] / hparams['num_player']))
    hparams['replay_buffer_size'] //= hparams['num_player']
    hparams['burn_in_frames'] //= hparams['num_player']

if not os.path.exists(hparams['work_dir']):
    os.makedirs(hparams['work_dir'])

logger_path = os.path.join(hparams['work_dir'], "train.log")
sys.stdout = common_utils.Logger(logger_path)
saver = common_utils.TopkSaver(hparams['work_dir'], 5)

common_utils.set_all_seeds(hparams['seed'])

explore_eps = utils.generate_explore_eps(
    hparams['act_base_eps'], hparams['act_eps_alpha'], hparams['num_t']
)
expected_eps = np.mean(explore_eps)
print("explore eps:", explore_eps)
print("avg explore eps:", np.mean(explore_eps))

if hparams['boltzmann_act']:
    boltzmann_beta = utils.generate_log_uniform(
        1 / hparams['max_t'], 1 / hparams['min_t'], hparams['num_t']
    )
    boltzmann_t = [1 / b for b in boltzmann_beta]
    print("boltzmann beta:", ", ".join(["%.2f" % b for b in boltzmann_beta]))
    print("avg boltzmann beta:", np.mean(boltzmann_beta))
else:
    boltzmann_t = []
    print("no boltzmann")

games = create_envs(hparams['num_thread'] * hparams['num_game_per_thread'])

in_dim = games[0].feature_size(hparams['sad'])
out_dim = games[0].num_action()
net_config = hparams['net']
net_config['in_dim'] = in_dim
net_config['out_dim'] = out_dim
if isinstance(in_dim, int):
    assert in_dim == 783
    net_config['priv_in_dim'] = in_dim - 125
    net_config['publ_in_dim'] = in_dim - 2 * 125
else:
    net_config['priv_in_dim'] = in_dim[1]
    net_config['publ_in_dim'] = in_dim[2]
agent = r2d2.R2D2Agent().to(hparams['train_device'])
agent.sync_target_with_online()
# for k, v in agent.state_dict().items():
#     print(f'{k} {str(v.dtype)[6:]}{list(v.shape)}')

if hparams['load_model'] and hparams['load_model'] != "None":
    if hparams['off_belief'] and hparams['belief_model'] != "None":
        belief_config = utils.get_train_config(hparams['belief_model'])
        hparams['load_model'] = belief_config["policy"]

    print("*****loading pretrained model*****")
    print(hparams['load_model'])
    utils.load_weight(agent.online_net, hparams['load_model'], hparams['train_device'])
    print("*****done*****")

# use clone bot for additional bc loss
if hparams['clone_bot'] and hparams['clone_bot'] != "None":
    clone_bot = utils.load_supervised_agent(hparams['clone_bot'], hparams['train_device'])
else:
    clone_bot = None

agent = agent.to(hparams['train_device'])
optim = torch.optim.Adam(agent.online_net.parameters(), lr=hparams['lr'], eps=hparams['eps'])
print(agent)
eval_agent = agent.clone(hparams['train_device'], {"vdn": False, "boltzmann_act": False})

replay_buffer = rela.RNNPrioritizedReplay(
    hparams['replay_buffer_size'],
    hparams['seed'],
    hparams['priority_exponent'],
    hparams['priority_weight'],
    hparams['prefetch'],
)

belief_model = None
if hparams['off_belief'] and hparams['belief_model'] != "None":
    print(f"load belief model from {hparams['belief_model']}")
    from belief_model import ARBeliefModel

    belief_devices = hparams['belief_device'].split(",")
    belief_config = utils.get_train_config(hparams['belief_model'])
    belief_model = []
    for device in belief_devices:
        belief_model.append(
            ARBeliefModel.load(
                hparams['belief_model'],
                device,
                5,
                hparams['num_fict_sample'],
                belief_config["fc_only"],
            )
        )

act_group = build_object_from_config(
    hparams, agent, explore_eps, boltzmann_t, replay_buffer, belief_model,
    cls="act_group.ActGroup"
)

context, threads = create_threads(
    hparams['num_thread'],
    hparams['num_game_per_thread'],
    act_group.actors,
    games,
)

act_group.start()
context.start()
while replay_buffer.size() < hparams['burn_in_frames']:
    print("warming up replay buffer:", replay_buffer.size())
    time.sleep(1)

print("Success, Done")
print("=======================")

frame_stat = dict()
frame_stat["num_acts"] = 0
frame_stat["num_buffer"] = 0

stat = common_utils.MultiCounter(hparams['work_dir'])
tachometer = utils.Tachometer()
stopwatch = common_utils.Stopwatch()

for epoch in range(hparams['num_epoch']):
    print("beginning of epoch: ", epoch)
    print(common_utils.get_mem_usage())
    tachometer.start()
    stat.reset()
    stopwatch.reset()

    for batch_idx in range(hparams['epoch_len']):
        num_update = batch_idx + epoch * hparams['epoch_len']
        if num_update % hparams['num_update_between_sync'] == 0:
            agent.sync_target_with_online()
        if num_update % hparams['actor_sync_freq'] == 0:
            act_group.update_model(agent)

        torch.cuda.synchronize()
        stopwatch.time("sync and updating")

        batch, weight = replay_buffer.sample(hparams['batchsize'], hparams['train_device'])
        stopwatch.time("sample data")

        loss, priority, online_q = agent.loss(batch, hparams['aux_weight'], stat)
        if clone_bot is not None and hparams['clone_weight'] > 0:
            bc_loss = agent.behavior_clone_loss(
                online_q, batch, hparams['clone_t'], clone_bot, stat
            )
            loss = loss + bc_loss * hparams['clone_weight']
        loss = (loss * weight).mean()
        loss.backward()

        torch.cuda.synchronize()
        stopwatch.time("forward & backward")

        g_norm = torch.nn.utils.clip_grad_norm_(
            agent.online_net.parameters(), hparams['grad_clip']
        )
        optim.step()
        optim.zero_grad()

        torch.cuda.synchronize()
        stopwatch.time("update model")

        replay_buffer.update_priority(priority)
        stopwatch.time("updating priority")

        stat["loss"].feed(loss.detach().item())
        stat["grad_norm"].feed(g_norm)
        stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

    count_factor = hparams['num_player'] if hparams['vdn'] else 1
    print("EPOCH: %d" % epoch)
    tachometer.lap(replay_buffer, hparams['epoch_len'] * hparams['batchsize'], count_factor)
    stopwatch.summary()
    stat.summary(epoch)

    eval_seed = (9917 + epoch * 999999) % 7777777
    eval_agent.load_state_dict(agent.state_dict())
    score, perfect, *_ = evaluate(
        [eval_agent for _ in range(hparams['num_player'])],
        1000,  # num game
        eval_seed,
        device=hparams['train_device'],
    )

    force_save_name = None
    if epoch > 0 and epoch % 100 == 0:
        force_save_name = "model_epoch%d" % epoch
    model_saved = saver.save(
        None, agent.online_net.state_dict(), score, force_save_name=force_save_name
    )
    print(
        "epoch %d, eval score: %.4f, perfect: %.2f, model saved: %s"
        % (epoch, score, perfect * 100, model_saved)
    )

    if clone_bot is not None:
        score, perfect, *_ = evaluate(
            [clone_bot] + [eval_agent for _ in range(hparams['num_player'] - 1)],
            1000,  # num game
            eval_seed,
        )
        print(f"clone bot score: {np.mean(score)}")

    if hparams['off_belief']:
        actors = common_utils.flatten(act_group.actors)
        success_fict = [actor.get_success_fict_rate() for actor in actors]
        print(
            "epoch %d, success rate for sampling ficticious state: %.2f%%"
            % (epoch, 100 * np.mean(success_fict))
        )
    print("==========")
