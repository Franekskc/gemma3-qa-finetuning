"""
Seed utilities for reproducibility.
"""

import random

import numpy as np
import torch

from gemmaqa.utils.logging import get_logger


def set_seed(seed: int):
    """
    Set seed for reproducibility across random, numpy, and torch.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    get_logger(__name__).info("Seed set", seed=seed)
