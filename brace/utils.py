"""Reproducibility, configuration and learning-rate schedule helpers."""
import math
import random

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_config(path, subset):
    """Return the full config and the train / model settings of one subset."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg["subsets"][subset]["train"], cfg["subsets"][subset]["model"]


def lr_factor(epoch, n_epochs):
    """Linear warmup over the first 10% of epochs, cosine decay over the rest."""
    warmup = 0.1 * n_epochs
    if epoch <= warmup:
        return epoch / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - math.ceil(warmup) - 1) / (0.9 * n_epochs)))
