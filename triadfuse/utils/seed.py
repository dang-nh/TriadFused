"""Reproducible seed utilities"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1337):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
