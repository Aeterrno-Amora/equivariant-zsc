# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import set_path

set_path.append_sys_path()

from common_utils.hparams import hparams
import rela
import hanalearn

assert rela.__file__.endswith(".so")
assert hanalearn.__file__.endswith(".so")

def create_envs(num_env, num_player=None, bomb=None, seed=None):
    if  num_player is None:
        num_player = hparams['num_player']
    if  bomb is None:
        bomb = hparams['bomb']
    if  seed is None:
        seed = hparams['seed']
    games = []
    for game_idx in range(num_env):
        params = {
            "players": str(hparams['num_player']),
            "seed": str(hparams['seed'] + game_idx),
            "bomb": str(bomb),
            "hand_size": str(hparams['hand_size']),
            "random_start_player": str(hparams['random_start_player']),
        }
        game = hanalearn.HanabiEnv(params, hparams['max_len'], False)
        games.append(game)
    return games


def create_threads(num_thread, num_game_per_thread, actors, games):
    context = rela.Context()
    threads = []
    for thread_idx in range(num_thread):
        envs = games[
            thread_idx * num_game_per_thread : (thread_idx + 1) * num_game_per_thread
        ]
        thread = hanalearn.HanabiThreadLoop(envs, actors[thread_idx], False)
        threads.append(thread)
        context.push_thread_loop(thread)
    print(
        "Finished creating %d threads with %d games and %d actors"
        % (len(threads), len(games), len(actors))
    )
    return context, threads
