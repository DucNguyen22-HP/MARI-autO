"""
Reproducibility Seed Utility
Responsible: SV3 (Đánh giá, Điều phối & Thí nghiệm đối chiếu seed 2)
"""
import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Sets global seeds across random, numpy, os, and PyTorch for scientific reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
