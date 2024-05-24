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
import pickle

import numpy as np
import torch

from common_utils.hparams import set_hparams, hparams
from act_group import ActGroup
from create import create_envs, create_threads
import common_utils
import rela
import utils
import belief_model


# def parse_args():
#     parser = argparse.ArgumentParser(description="train belief model")
#     parser.add_argument("--work_dir", type=str, default="exps/dev_belief")
#     parser.add_argument("--load_model", type=int, default=0)
#     parser.add_argument("--seed", type=int, default=10001)
#     parser.add_argument("--hid_dim", type=int, default=512)
#     parser.add_argument("--fc_only", type=int, default=0)
#     parser.add_argument("--train_device", type=str, default="cuda:0")
#     parser.add_argument("--act_device", type=str, default="cuda:1")

#     # load policy config
#     parser.add_argument("--policy", type=str, default="")
#     parser.add_argument("--explore", type=int, default=1)
#     parser.add_argument("--rand", type=int, default=0)
#     parser.add_argument("--clone_bot", type=int, default=0)

#     # optimization/training settings
#     parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
#     parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
#     parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
#     parser.add_argument("--batchsize", type=int, default=128)
#     parser.add_argument("--num_epoch", type=int, default=1000)
#     parser.add_argument("--epoch_len", type=int, default=1000)

#     # replay buffer settings
#     parser.add_argument("--burn_in_frames", type=int, default=80000)
#     parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
#     parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

#     # thread setting
#     parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
#     parser.add_argument("--num_game_per_thread", type=int, default=20)

#     # load from dataset setting
#     parser.add_argument("--dataset", type=str, default="")
#     parser.add_argument("--num_player", type=int, default=2)
#     parser.add_argument("--inf_data_loop", type=int, default=0)
#     parser.add_argument("--max_len", type=int, default=80)
#     parser.add_argument("--shuffle_color", type=int, default=0)

#     args = parser.parse_args()
#     return args


def create_rl_context():
    agent_overwrite = {
        "vdn": False,
        "device": hparams['train_device'],  # batch runner will create copy on act device
        "uniform_priority": True,
    }

    if hparams['clone_bot']:
        agent = utils.load_supervised_agent(hparams['policy'], hparams['train_device'])
        cfgs = {
            "act_base_eps": 0.1,
            "act_eps_alpha": 7,
            "num_game_per_thread": 80,
            "num_player": 2,
            "train_bomb": 0,
            "max_len": 80,
            "sad": 0,
            "shuffle_color": 0,
            "hide_action": 0,
            "multi_step": 1,
            "gamma": 0.999,
        }
    else:
        agent, cfgs = utils.load_agent(hparams['policy'], agent_overwrite)

    assert cfgs["shuffle_color"] == False
    assert hparams['explore']

    replay_buffer = rela.RNNPrioritizedReplay(
        hparams['replay_buffer_size'],
        hparams['seed'],
        1.0,  # priority exponent
        0.0,  # priority weight
        hparams['prefetch'],
    )

    if hparams['rand']:
        explore_eps = [1]
    elif hparams['explore']:
        # use the same exploration config as policy learning
        explore_eps = utils.generate_explore_eps(
            cfgs["act_base_eps"], cfgs["act_eps_alpha"], cfgs["num_game_per_thread"]
        )
    else:
        explore_eps = [0]

    expected_eps = np.mean(explore_eps)
    print("explore eps:", explore_eps)
    print("avg explore eps:", np.mean(explore_eps))
    if hparams['clone_bot'] or not agent.boltzmann:
        print("no boltzmann act")
        boltzmann_t = []
    else:
        boltzmann_beta = utils.generate_log_uniform(
            1 / cfgs["max_t"], 1 / cfgs["min_t"], cfgs["num_t"]
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
        print("boltzmann beta:", ", ".join(["%.2f" % b for b in boltzmann_beta]))
        print("avg boltzmann beta:", np.mean(boltzmann_beta))

    games = create_envs( #TODO: cfgs
        hparams['num_thread'] * hparams['num_game_per_thread'],
        cfgs["num_player"], # can be different?
        cfgs["max_len"], # can be different?
    )

    act_group = ActGroup(
        agent,
        explore_eps,
        boltzmann_t,
        replay_buffer,
        None,  # belief_model
        hparams['act_device'],
        hparams['seed'],
        hparams['num_thread'],
        hparams['num_game_per_thread'],
        cfgs["num_player"],
        False, # iql instead of vdn
        cfgs["sad"],
        cfgs["shuffle_color"] if not hparams['rand'] else False,
        cfgs["hide_action"],
        False,  # not trinary, need full hand for prediction
        cfgs["multi_step"],  # not used
        cfgs["max_len"],
        cfgs["gamma"],  # not used
        False,  # turn off off-belief rewardless of how it is trained
    )

    context, threads = create_threads(
        hparams['num_thread'],
        hparams['num_game_per_thread'],
        act_group.actors,
        games,
    )
    return agent, cfgs, replay_buffer, games, act_group, context, threads


def create_sl_context():
    games = pickle.load(open(hparams['dataset'], "rb"))
    print(f"total num game: {len(games)}")
    if hparams['shuffle_color']:
        # to make data generation speed roughly the same as consumption
        hparams['num_thread'] = 10
        hparams['inf_data_loop'] = 1

    if hparams['replay_buffer_size'] < 0:
        hparams['replay_buffer_size'] = len(games) * hparams['num_player']
    if hparams['burn_in_frames'] < 0:
        hparams['burn_in_frames'] = len(games) * hparams['num_player']

    # priority not used
    priority_exponent = 1.0
    priority_weight = 0.0
    replay_buffer = rela.RNNPrioritizedReplay(
        hparams['replay_buffer_size'],
        hparams['seed'],
        priority_exponent,
        priority_weight,
        hparams['prefetch'],
    )
    data_gen = hanalearn.CloneDataGenerator(
        replay_buffer,
        hparams['num_player'],
        hparams['max_len'],
        hparams['shuffle_color'],
        False,
        hparams['num_thread'],
    )
    game_params = {
        "players": str(hparams['num_player']),
        "random_start_player": "0",
        "bomb": "0",
    }
    data_gen.set_game_params(game_params)
    for i, g in enumerate(games):
        data_gen.add_game(g["deck"], g["moves"])
        if (i + 1) % 10000 == 0:
            print(f"{i+1} games added")

    return data_gen, replay_buffer


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    set_hparams()

    if not os.path.exists(hparams['work_dir']):
        os.makedirs(hparams['work_dir'])

    logger_path = os.path.join(hparams['work_dir'], "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(hparams['work_dir'], 2)

    common_utils.set_all_seeds(hparams['seed'])
    pprint.pprint(hparams)

    if hparams['dataset'] is None or len(hparams['dataset']) == 0:
        (
            agent,
            cfgs,
            replay_buffer,
            games,
            act_group,
            context,
            threads,
        ) = create_rl_context()
        act_group.start()
        context.start()
    else:
        data_gen, replay_buffer = create_sl_context()
        data_gen.start_data_generation(hparams['inf_data_loop'], hparams['seed'])
        # only for getting feature size
        games = create_envs(1)
        cfgs = {"sad": False}

    while replay_buffer.size() < hparams['burn_in_frames']:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    print("Success, Done")
    print("=======================")

    if hparams['load_model']:
        belief_config = utils.get_train_config(cfgs["belief_model"])
        print("load belief model from:", cfgs["belief_model"])
        model = belief_model.ARBeliefModel.load(
            cfgs["belief_model"],
            hparams['train_device'],
            5,
            0,
            belief_config["fc_only"],
        )
    else:
        model = belief_model.ARBeliefModel(
            hparams['train_device'],
            games[0].feature_size(cfgs["sad"])[1],
            hparams['hid_dim'],
            5,  # hand_size
            25,  # bits per card
            0,  # num_sample
            fc_only=hparams['fc_only'],
        ).to(hparams['train_device'])

    optim = torch.optim.Adam(model.parameters(), lr=hparams['lr'], eps=hparams['eps'])

    stat = common_utils.MultiCounter(hparams['work_dir'])
    tachometer = utils.Tachometer()
    for epoch in range(hparams['num_epoch']):
        print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()

        for batch_idx in range(hparams['epoch_len']):
            batch, weight = replay_buffer.sample(hparams['batchsize'], hparams['train_device'])
            assert weight.max() == 1
            loss, xent, xent_v0, _ = model.loss(batch)
            loss = loss.mean()
            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['grad_clip'])
            optim.step()
            optim.zero_grad()
            replay_buffer.update_priority(torch.Tensor())

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)
            stat["xent_pred"].feed(xent.detach().mean().item())
            stat["xent_v0"].feed(xent_v0.detach().mean().item())

        print("EPOCH: %d" % epoch)

        if hparams['dataset'] is None or len(hparams['dataset']) == 0:
            scores = [g.last_episode_score() for g in games]
            print("mean score: %.2f" % np.mean(scores))

        count_factor = 1
        tachometer.lap(replay_buffer, hparams['epoch_len'] * hparams['batchsize'], count_factor)

        force_save_name = None
        if epoch > 0 and epoch % 100 == 0:
            force_save_name = "model_epoch%d" % epoch
        saver.save(
            None,
            model.state_dict(),
            -stat["loss"].mean(),
            True,
            force_save_name=force_save_name,
        )
        stat.summary(epoch)
        print("===================")
