#!/usr/bin/env bash

CONDA_NAME=cs394r

conda create -n ${CONDA_NAME} python=3.7 -y;
conda activate ${CONDA_NAME}

conda install cudatoolkit=10.1
pip install torch===1.4.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tianshou==0.4.5
pip install open_spiel
pip install gym==0.19.0
pip install opencv-python

# extra commands to run tianshou's IQN example with atari
pip install gym\[atari\]
pip install envpool
