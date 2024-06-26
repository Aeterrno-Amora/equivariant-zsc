# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import sys
import time
import logging
import torch
import torch.nn.functional as F
import pprint

import set_path

set_path.append_sys_path()
from create import create_envs
import r2d2
import common_utils
import utils
import rela
import hanalearn

from common_utils.hparams import set_hparams, hparams
from supervised_model import SupervisedAgent
from eval import evaluate

torch.backends.cudnn.benchmark = True

set_hparams()


def compute_loss(pred_logits, legal_move, gt_a, mask):
    seq_len, bsize, num_actions = pred_logits.shape
    assert pred_logits.size() == legal_move.size()
    pred_logits = pred_logits - (1 - legal_move) * 1e10
    pred_logits = pred_logits.reshape(-1, num_actions)
    gt_a = gt_a.flatten()
    loss = F.cross_entropy(pred_logits, gt_a, reduction="none")
    loss = loss.view(seq_len, bsize)
    loss = (loss * mask).sum(0).mean()
    return loss


def train(
    model,
    device,
    optim,
    replay_buffer,
    batchsize,
    num_batch,
    grad_clip,
    stat,
    stopwatch,
):
    for i in range(num_batch):
        batch, weight = replay_buffer.sample(batchsize, device)
        priv_s = batch.obs["priv_s"]
        publ_s = batch.obs["publ_s"]
        legal_move = batch.obs["legal_move"]
        action = batch.action["a"]
        mask = torch.arange(0, priv_s.size(0), device=batch.seq_len.device)
        mask = (mask.unsqueeze(1) < batch.seq_len.unsqueeze(0)).float()
        if stopwatch is not None:
            torch.cuda.synchronize()
            stopwatch.time("sample data")

        logits, _ = model(priv_s, publ_s, None)
        loss = compute_loss(logits, legal_move, action, mask)
        loss.backward()
        if stopwatch is not None:
            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

        g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        optim.zero_grad()
        replay_buffer.update_priority(weight.cpu())
        if stopwatch is not None:
            stopwatch.time("update model")

        stat["loss"].feed(loss.item())
        stat["grad_norm"].feed(g_norm)
    return


def train_rl(
    agent,
    device,
    optim,
    replay_buffer,
    batchsize,
    num_batch,
    grad_clip,
    epoch,
    num_update_between_sync,
    aux_weight,
    stat,
    stopwatch,
):
    for i in range(num_batch):
        num_update = i + epoch * hparams['epoch_len']
        if num_update % num_update_between_sync == 0:
            agent.sync_target_with_online()

        batch, weight = replay_buffer.sample(batchsize, device)
        if stopwatch is not None:
            torch.cuda.synchronize()
            stopwatch.time("sample data")

        loss, priority, _ = agent.loss(batch, aux_weight, 0, stat)
        loss = loss.mean()
        loss.backward()

        if stopwatch is not None:
            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

        g_norm = torch.nn.utils.clip_grad_norm_(
            agent.online_net.parameters(), grad_clip
        )
        optim.step()
        optim.zero_grad()
        replay_buffer.update_priority(weight.cpu())

        if stopwatch is not None:
            stopwatch.time("update model")

        stat["loss"].feed(loss.item())
        stat["grad_norm"].feed(g_norm)
    return


def create_data_generator(
    games,
    num_player,
    num_thread,
    inf_data_loop,
    max_len,
    shuffle_color,
    replay_buffer_size,
    prefetch,
    seed,
):
    print(f"total num game: {len(games)}")
    # priority not used
    priority_exponent = 1.0
    priority_weight = 0.0
    replay_buffer = rela.RNNPrioritizedReplay(
        replay_buffer_size,
        seed,
        priority_exponent,
        priority_weight,
        prefetch,
    )
    data_gen = hanalearn.CloneDataGenerator(
        replay_buffer, num_player, max_len, shuffle_color, True, num_thread
    )
    game_params = {
        "players": str(num_player),
        "random_start_player": "0", # TODO: why hardcoded?
        "bomb": str(hparams['bomb']),
    }
    data_gen.set_game_params(game_params)
    for i, g in enumerate(games):
        data_gen.add_game(g["deck"], g["moves"])
        if (i + 1) % 10000 == 0:
            print(f"{i+1} games added")

    return data_gen, replay_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="exps/clone_bot0")
    parser.add_argument("--method", type=str, default="sl", help="sl/rl")
    # rl setting
    parser.add_argument("--multi_step", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--aux_weight", type=float, default=0)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)
    # data info
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--inf_data_loop", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e5))
    parser.add_argument("--prefetch", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    # network config
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--net", type=str, default="lstm")
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument("--num_lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--load_ckpt", type=str, default=None)
    # optim
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--grad_clip", type=float, default=5)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_epoch", type=int, default=1000)
    # others
    parser.add_argument("--dev", type=int, default=0)

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s]%(levelname)s: %(message)s",
    )
    args = parser.parse_args()
    if hparams['shuffle_color']:
        # to make data generation speed roughly the same as consumption
        hparams['num_thread'] = 10
        hparams['inf_data_loop'] = 1

    os.makedirs(hparams['save_dir'], exist_ok=True)
    sys.stdout = common_utils.Logger(os.path.join(hparams['save_dir'], "train.log"))
    saver = common_utils.TopkSaver(hparams['save_dir'], 3)

    games = pickle.load(open(hparams['dataset'], "rb"))

    if hparams['replay_buffer_size'] < 0:
        hparams['replay_buffer_size'] = len(games) * hparams['num_player']

    # create data generator
    data_gen, replay_buffer = create_data_generator(
        games,
        hparams['num_player'],
        hparams['num_thread'],
        hparams['inf_data_loop'],
        hparams['max_len'],
        hparams['shuffle_color'],
        hparams['replay_buffer_size'],
        hparams['prefetch'],
        hparams['seed'],
    )
    data_gen.start_data_generation(hparams['inf_data_loop'], hparams['seed'])

    # create network and optim
    game = create_envs(1)[0]
    _, priv_in_dim, publ_in_dim = game.feature_size(False)
    num_action = game.num_action()
    if hparams['method'] == "sl":
        net = SupervisedAgent(
            hparams['train_device'],
            priv_in_dim,
            publ_in_dim,
            hparams['rnn_hid_dim'],
            num_action,
            hparams['num_lstm_layers'],
            hparams['net'],
            hparams['dropout'],
        ).to(hparams['train_device'])
        optim = torch.optim.Adam(net.parameters(), lr=hparams['lr'], eps=hparams['eps'])
    elif hparams['method'] == "rl":
        in_dim = game.feature_size(False)
        net_config = hparams['net']
        net_config['in_dim'] = in_dim
        net_config['out_dim'] = num_action
        if isinstance(in_dim, int):
            assert in_dim == 783
            net_config['priv_in_dim'] = in_dim - 125
            net_config['publ_in_dim'] = in_dim - 2 * 125
        else:
            net_config['priv_in_dim'] = in_dim[1]
            net_config['publ_in_dim'] = in_dim[2]
        agent = r2d2.R2D2Agent().to(hparams['train_device'])
        agent.sync_target_with_online()
        optim = torch.optim.Adam(
            agent.online_net.parameters(), lr=hparams['lr'], eps=hparams['eps']
        )
        net = agent

    # save extra values for easy loading
    hparams['priv_in_dim'] = priv_in_dim
    hparams['publ_in_dim'] = publ_in_dim
    hparams['num_action'] = num_action
    pprint.pprint(vars(args))

    if hparams['load_ckpt']:
        checkpoint = torch.load(root_path / hparams['load_ckpt'])
        print(f"Load checkpoint at {root_path / hparams['load_ckpt']}")
        net.load_state_dict(checkpoint)

    speed = 0
    total_t = time.time()
    while replay_buffer.size() < hparams['replay_buffer_size']:
        prev_replay_size = replay_buffer.size()
        print(f"generating data: {prev_replay_size}, speed: {speed}/s")
        print(common_utils.get_mem_usage())
        time.sleep(1)
        new_size = replay_buffer.size()
        speed = new_size - prev_replay_size
    total_t = time.time() - total_t
    print(f"total waiting time: {total_t}")

    stat = common_utils.MultiCounter(None)
    stopwatch = common_utils.Stopwatch()
    tachometer = utils.Tachometer()
    best_eval = 0
    for epoch in range(hparams['num_epoch']):
        stat.reset()
        stopwatch.reset()
        tachometer.start()

        if hparams['method'] == "sl":
            train(
                net,
                hparams['train_device'],
                optim,
                replay_buffer,
                hparams['batchsize'],
                hparams['epoch_len'],
                hparams['grad_clip'],
                stat,
                stopwatch,
            )
        elif hparams['method'] == "rl":
            train_rl(
                agent,
                hparams['train_device'],
                optim,
                replay_buffer,
                hparams['batchsize'],
                hparams['epoch_len'],
                hparams['grad_clip'],
                epoch,
                hparams['num_update_between_sync'],
                hparams['aux_weight'],
                stat,
                stopwatch,
            )

        print(f"Epoch {epoch}:")
        print(common_utils.get_mem_usage())
        tachometer.lap(replay_buffer, hparams['epoch_len'] * hparams['batchsize'], 1)
        stopwatch.summary()
        stat.summary(epoch)

        net.train(False)
        score, perfect, *_ = evaluate(
            [net for _ in range(hparams['num_player'])],
            10000,  # num game
            1,  # seed
            device=hparams['train_device'],
        )
        net.train(True)
        print(f"epoch {epoch}, eval score: {score:5f}, perfect: {100 * perfect:.2f}%")
        model_saved = saver.save(None, net.state_dict(), score)
        print("==========================================")
    replay_buffer.terminate()
    data_gen.terminate()
