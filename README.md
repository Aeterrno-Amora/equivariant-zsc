# Equivariant Networks for Zero-Shot Coordination
## Introduction

This repo contains the implementation of the EQC architecture proposed in Equivariant Networks for Zero-Shot Coordination, NeurIPS 2022.

This repo is largely based off the [Off-Belief Learning repo](https://github.com/facebookresearch/off-belief-learning) and contains the implementation of the algorithm proposed in [Off-Belief Learning, ICML 2021](https://arxiv.org/pdf/2103.04000.pdf). As such, much of the README from that repo is the same here.

## Environment Setup

We have been using `python-3.12.2`, `pytorch-2.2.2`, `cuda-12.0`, and `cudnn-v8.9.2` in
our development environment. We have not tested it extensively in
other environment configurations but it may also work. You will need
to change the pybind submodule to the same version as the one used by
your pytorch, which is detailed in later section. We also use
conda/miniconda to manage environments.

```shell
conda create -n hanabi python=3.12
conda activate hanabi

# install pytorch
# the code was developed with pytorch 2.2.2 but newer versions may also work
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# install other dependencies
pip install psutil

# install a newer cmake if the current version is < 3.24
conda install -c conda-forge cmake
```

To help cmake find the proper libraries (e.g. libtorch), please either
add the following lines to your `.bashrc`, or add it to a separate file
and `source` it before you start working on the project.

```shell
# activate the conda environment
conda activate hanabi

# set path
CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CPATH=${CONDA_PREFIX}/include:${CPATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# avoid tensor operation using all cpu cores
export OMP_NUM_THREADS=1
```

If you use *another version of pytorch* instead of 2.2.2, you may need
to check out the pybind module to use the corresponding version (the
version can be found at pybind11 row [here](https://github.com/pytorch/pytorch/tree/master/third_party)):
```
cd third_party/pybind11
git checkout v2.11
cd ../..
```

Finally, to compile this repo:

```shell
# under project root
mkdir build
cd build
cmake ..
make -j10
```

## Code Structure

For an overview of the training infrastructure, please refer to Figure 5 of the
[Off-Belief Learning] (https://arxiv.org/pdf/2103.04000.pdf) paper.

`hanabi-learning-environment` is a modified version of the original
[HLE from Deepmind](https://github.com/deepmind/hanabi-learning-environment).

Notable modifications includes:

1) Card knowledge part of the observation encoding is changed to
v0-belief, i.e.  card knowledge normalized by the remaining public
card count.

2) Functions to reset the game state with sampled hands.

`rela` (REinforcement Learning Assemly) is a set of tools for
efficient batched neural network inference written in C++ with
multi-threading.

`rlcc` implements the core of various algorithms. For example, the
logic of fictitious transitions are implemented in `r2d2_actor.cc`.
It also contains implementations of baselines such as other-play, VDN
and IQL.

`pyhanabi` is the main entry point of the repo. It contains implementations for
Q-network, recurrent DQN training, belief network and training, as well as some tools
to analyze trained models.

`bot` is the human-ai evaluation using web interface. It may be unsupported in the current brunch.

## Run the Code

Please refer to the README in pyhanabi for detailed instruction on how to train a model.

## Download Models

To download the trained models used in the paper, go to `models` folder and run

```shell
sh download.sh
```

Due to agreement with BoardGameArena and Facebook policies, we are
unable to release the "Clone Bot" models trained on the game data nor
the datasets themselves.

## Copyright
Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

This source code is licensed under the license found in the LICENSE
file in the root directory of this source tree.
