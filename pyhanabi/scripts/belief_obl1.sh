# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python train_belief.py \
       --work_dir exps/belief_obl1 \
       --num_thread 40 \
       --num_game_per_thread 40 \
       --batchsize 128 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --hid_dim 512 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 200 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --explore 1 \
       --policy exps/obl1/model0.pthw \
       --seed 2254257 \
       --num_player 2 \
       --shuffle_color 0 \
       --load_model 1 \
