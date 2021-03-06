#!/usr/bin/env bash

CONDA_NAME=cs394r;

conda create -n ${CONDA_NAME} python=3.7 -y;
conda activate ${CONDA_NAME};

conda install cudatoolkit=10.1;
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tianshou==0.4.7;
pip install --upgrade open_spiel;
pip install gym==0.23.0;
pip install opencv-python matplotlib tqdm pettingzoo==1.17.0;

# extra commands to run tianshou's IQN example with atari
# pip install gym\[atari\];
pip install ale_py
pip install envpool;

pip install rlcard;
pip install pygame;
pip install seaborn;
pip install jupyter;
pip install ray;

