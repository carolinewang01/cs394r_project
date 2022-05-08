import torch
import numpy as np


def set_seed(seed, envs:list):
    np.random.seed(seed)
    torch.manual_seed(seed)
    for env in envs:
        env.seed(seed)
