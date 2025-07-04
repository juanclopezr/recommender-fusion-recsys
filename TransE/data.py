from torch.utils import data
from typing import Dict, Tuple
import numpy as np


class Dataset(data.Dataset):
    """Dataset implementation for handling edges.npy."""

    def __init__(self, data_path: str):
        self.data = self.numpy_read(data_path)

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)
    
    def __total_entities__(self):
        """Extracts the total number of nodes"""
        return len(set(self.data.flatten()))

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        head_id, tail_id = self.data[index]
        return head_id, 0, tail_id
    
    @staticmethod
    def numpy_read(path: str):
        with open(path, 'rb') as file:
            data = np.load(file)
        return data
