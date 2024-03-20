import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from typing import List

class DataPrepper:
    def __init__(self) -> None:
        pass

    def build_dataset(self, ds : Dataset, exclude_labels: List[int], batch_size:int=64):
        filtered = [sample for sample in ds if sample[1] not in exclude_labels]
        filtered_v, filtered_l = zip(*filtered)
        filtered_v = torch.stack(filtered_v)
        filtered_l = torch.tensor(filtered_l)
        
        return DataLoader(
            TensorDataset(filtered_v, filtered_l), 
            batch_size=batch_size
        )
