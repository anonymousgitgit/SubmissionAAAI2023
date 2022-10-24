from functools import reduce
from collections import defaultdict
import numpy as np
import torch
import dgl
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from data.tokenizer import split_identifier_into_parts
from modules.utils import align, make_src_map
from constants import PAD_IDX

def train_collate():
    def fn(batch):
        in_token_ids, out_token_ids, outputs = list(zip(*batch))
        in_token_ids = list(map(torch.Tensor, in_token_ids))
        out_token_ids = list(map(torch.Tensor, out_token_ids))
        return in_token_ids, out_token_ids, outputs
    return fn

def make_data_loader(dataset, batch_size, task = 'classification', shuffle = False, training = True, num_workers = 0, *, tokenizer = None):
    return DataLoader(dataset, batch_size = batch_size, collate_fn = (train_collate)(), shuffle = shuffle, num_workers = num_workers)
