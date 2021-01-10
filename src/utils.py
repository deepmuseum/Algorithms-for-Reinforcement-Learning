import torch
import numpy as np


def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
