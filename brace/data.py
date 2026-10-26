"""CMU-MOSI / CMU-MOSEI loader for the fixed-missing protocol.

A run keeps one fixed modality subset observed in every split; the other
modalities are zeroed out.
"""
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .constants import LETTERS, MODALITIES


def present_mask(subset):
    observed = {LETTERS[ch] for ch in subset}
    return tuple(m in observed for m in MODALITIES)


def build_loaders(path, subset, batch_size):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    present = present_mask(subset)
    loaders = {}
    for split in ("train", "valid", "test"):
        data = blob[split]
        features = []
        for modality, keep in zip(MODALITIES, present):
            x = np.asarray(data[modality], dtype=np.float32)
            x[x == -np.inf] = 0
            features.append(torch.from_numpy(x if keep else np.zeros_like(x)))
        labels = torch.from_numpy(np.asarray(data["regression_labels"], dtype=np.float32).reshape(-1, 1))
        loaders[split] = DataLoader(
            TensorDataset(*features, labels),
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
    return loaders, present
