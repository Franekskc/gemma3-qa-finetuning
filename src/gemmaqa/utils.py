"""
Utility helpers
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed set to %d", seed)
