from json import load
from typing import Callable, Optional, List
from numpy import dtype
from torch_geometric.data import InMemoryDataset, Data
import torch
import os.path as osp
import pickle
from torch_geometric.utils import to_undirected
import numpy as np

def load_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


class DistillData(InMemoryDataset):
    def __init__(self, root: str, name: str, transform = None, pre_transform = None):
        self.name = name
        
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def raw_file_names(self) -> List[str]:
        names = ['label_map.pkl', 'train_ids.pkl', 'test_ids.pkl', 'real_feature.pkl', 're_id.pkl']
        return names
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def __repr__(self) -> str:
        return f'{self.name}()'

