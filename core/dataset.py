from torch.utils.data import Dataset
import torch
from torch import nn

import pandas as pd
import json
from pathlib import Path
import random

class MolData(Dataset):
    def __init__(self, data_path):
        tensor = torch.load(data_path)
        self.props = tensor[:, :-1]
        self.scores = tensor[:, -1]
        self.stats = self.get_stats()

    def get_stats(self):
        """
        Get the mean and std of the values.
        """

        stats = {
            "prop": {},
            "score": {}
        }

        for type_, values in zip(stats, [self.props, self.scores]):
            mean = values.mean(0)
            std = values.std(0)
            stats[type_]["mean"] = mean
            stats[type_]["std"] = std

        return stats

    def normalize(self, values, type_):
        """
        Rescale values so their mean = 0 and std = 1.
        """

        mean, std = self.stats[type_]["mean"], self.stats[type_]["std"]
        normalized = (values - mean) / std

        return normalized

    def __len__(self):
        return len(self.props)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        normalized_score = self.normalize(self.scores[idx], "score")
        normalized_props = self.normalize(self.props[idx], "prop")
        normalized_props[2:] = self.props[idx][2:]

        return normalized_props, normalized_score
