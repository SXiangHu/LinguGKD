import json
import numpy as np
from torch_geometric.utils import k_hop_subgraph, subgraph
from datasets import Dataset, DatasetDict, load_dataset
from os import path
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader, Data
from sklearn.model_selection import train_test_split
import torch
from graph_utils import bfs_paths, node2json
from transformers import PreTrainedTokenizer
from man_utils import save_json, load_json
from tqdm import tqdm


DATASET_MAPPING = {
    'Cora': Planetoid,
    'CiteSeer': Planetoid,
    'PubMed': Planetoid
}


def load_graph(args):
    dataset = DATASET_MAPPING[args.dataset_name](
        args.graph_data_dir, args.dataset_name)
    return dataset


def split_dataset(dataset: Data, split: list):
    train_idx, test_idx = train_test_split(
        range(dataset.num_nodes), test_size=split[2], random_state=42)
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=split[1], random_state=42)

    # 创建训练集、验证集、测试集掩码
    train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True

    train_dataset = dataset.subgraph(train_mask)
    valid_dataset = dataset.subgraph(valid_mask)
    test_dataset = dataset.subgraph(test_mask)

    return train_dataset, valid_dataset, test_dataset
