from functools import reduce
from collections import defaultdict
import numpy as np
import torch
import dgl
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from data.tokenizer import split_identifier_into_parts
from constants import PAD_IDX

def _pad_batch_2D(batch, value = 0):
    max_batch = max([len(x) for x in batch])
    batch = [n + [value] * (max_batch - len(n)) for n in batch]
    batch = np.asarray(batch)
    return batch

def _pad_batch_3D(batch, value = 0):
    max_2nd_D = max([len(x) for x in batch])
    max_3rd_D = max([len(c) for n in batch for c in n])
    batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
    batch = [[c + [value] * (max_3rd_D - len(c)) for c in sample] for sample in batch]
    batch = np.asarray(batch)
    return batch

def _pad_batch_4D(batch, value = 0):
    max_2nd_D = max([len(x) for x in batch])
    max_3rd_D = max([len(c) for n in batch for c in n])
    max_4th_D = max([len(s) for n in batch for c in n for s in c] or [value])
    batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
    batch = [[c + ([[]] * (max_3rd_D - len(c))) for c in sample] for sample in batch]
    batch = [[[s + [value] * (max_4th_D - len(s)) for s in c] for c in sample] for sample in batch]
    batch = np.asarray(batch)
    return batch

def make_batch(batch_data):
    batch_node_index = []
    batch_node_type_id = []
    batch_node_sub_tokens_id = []
    batch_node_token = []
    batch_node_height = []

    batch_children_index = []
    # batch_children_node_type_id = []
    # batch_children_node_sub_tokens_id = []
    # batch_children_node_token = []

    batch_size = []
    tree_index = []
    for tree_data in batch_data:
        # for Sinusoid encoding
        batch_node_index.append((np.array(tree_data["node_index"]) + 1).tolist())
        batch_node_type_id.append(tree_data["node_type_id"])
        batch_node_sub_tokens_id.append(tree_data["node_sub_token_ids"])
        batch_node_token.append(tree_data["node_token"])
        batch_node_height.append(tree_data['node_height'])

        batch_children_index.append(tree_data["children_index"])
        # batch_children_node_type_id.append(tree_data["children_node_type_id"])
        # batch_children_node_sub_tokens_id.append(tree_data["children_node_sub_token_ids"])
        # batch_children_node_token.append(tree_data["children_node_token"])

        batch_size.append(tree_data["size"])
        tree_index.append(tree_data['tree_index'])
    
    batch_node_height = _pad_batch_2D(batch_node_height)
    # [[]]
    batch_node_index = _pad_batch_2D(batch_node_index)
    # [[]]
    batch_node_type_id = _pad_batch_2D(batch_node_type_id)
    # [[[]]]
    batch_node_sub_tokens_id = _pad_batch_3D(batch_node_sub_tokens_id)
    # [[[]]]
    batch_children_index = _pad_batch_3D(batch_children_index)
    # [[[]]]
    # batch_children_node_type_id = _pad_batch_3D(batch_children_node_type_id)    
    # [[[[]]]]
    # batch_children_node_sub_tokens_id = _pad_batch_4D(batch_children_node_sub_tokens_id, -1)
    # if batch_children_node_sub_tokens_id.ndim < 4:
        # batch_children_node_sub_tokens_id = np.expand_dims(batch_children_node_sub_tokens_id, -1)


    batch_obj = {
        'batch_tree_index': tree_index,
        'batch_node_height': batch_node_height,
        "batch_node_index": batch_node_index,
        "batch_node_type_id": batch_node_type_id,
        "batch_node_sub_tokens_id": batch_node_sub_tokens_id,
        "batch_children_index": batch_children_index,
        # "batch_children_node_type_id": batch_children_node_type_id,
        # "batch_children_node_sub_tokens_id": batch_children_node_sub_tokens_id,
        "batch_tree_size": batch_size
    }
    return batch_obj

def get_ast_index(vocab_obj, ast_nodes, num_nodes):
    data = {}
    offset = 0
    for ast, num_node in zip(ast_nodes, num_nodes):
        for node, info in ast.items():
            data[node + offset] = {
                'node_type_index': torch.tensor(vocab_obj.get_id_from_node_type(info['node_type'])),
                'node_sub_token_ids': torch.tensor(list(map(vocab_obj.get_id_from_sub_token, split_identifier_into_parts(info['node_token']))))
            }
        offset += num_node
    return data

def adjust_tree_index(tree_data, num_nodes):
    offset = 0
    last_stmt_indices = []
    for i, trees in enumerate(tree_data):
        max_stmt_index = 0
        for tree in trees:
            tree['tree_index'] += offset
            max_stmt_index = max(max_stmt_index, tree['tree_index'])
        last_stmt_indices.append(max_stmt_index)
        offset += num_nodes[i]
    return last_stmt_indices
def group_by_size(tree_data):
    buckets = defaultdict(list)
    for tree in tree_data:
        size = tree['size']
        if size < 10:
            buckets[size].append(tree)
        else:
            idx = (size // 5) * 5
            buckets[idx].append(tree)
    return dict(buckets)

def concat_degrees(degrees, num_node_types):
    data = defaultdict(list)
    node_types = ['node'] if num_node_types == 1 else ['ast_node', 'stmt_node']
    for item in degrees:
        for node_type in node_types:
            data[node_type] = np.concatenate((data[node_type], item[node_type]))
    data = dict(data)
    tensor_data = {}
    for node_type in node_types:
        tensor_data[node_type] = torch.Tensor(data[node_type])
    return tensor_data
    

def clone_train_collate(vocab_obj, dgl_format, num_node_types, task, triplet):
    def fn(batch):
        if triplet:
            anchor_samples, pos_samples, neg_samples, labels = list(zip(*batch))
            merge_batch = anchor_samples + pos_samples + neg_samples
        else:
            merge_batch, labels = list(zip(*batch))
        if dgl_format:
            if num_node_types == 1:
                if task == 'clone':
                    _ast_nodes, _tree_data, graphs, in_degrees, out_degrees = list(zip(*merge_batch))
                num_nodes = [g.num_nodes() for g in graphs]
                last_stmt_indices = adjust_tree_index(_tree_data, num_nodes)
                ast_node_index = get_ast_index(vocab_obj, _ast_nodes, num_nodes)
            elif num_node_types == 2:
                _ast_nodes, _tree_data, graphs, in_degrees, out_degrees, num_nodes = list(zip(*merge_batch))
                num_ast_nodes = [x['ast_nodes'] for x in num_nodes]
                num_stmt_nodes = [x['stmt_nodes'] for x in num_nodes]
                last_stmt_indices = adjust_tree_index(_tree_data, num_stmt_nodes)
                ast_node_index = get_ast_index(vocab_obj, _ast_nodes, num_ast_nodes)
            tree_data = reduce(list.__add__, _tree_data)
            buckets =  group_by_size(tree_data)
            batch_buckets = {k: make_batch(v) for k, v in buckets.items()}
            # batch_tree = make_batch(tree_data)
            in_degrees = concat_degrees(in_degrees, num_node_types)
            batch_tensor_buckets = {}
            for size, batch in batch_buckets.items():
                batch_tensor_buckets[size] = {}
                for k, v in batch.items():
                    batch_tensor_buckets[size][k] = torch.Tensor(v) if k != 'batch_tree_index' else v
              
            # batch_tensor_tree = dict({k: torch.Tensor(v) if k != 'batch_tree_index' else v for k, v in batch_tree.items()})
            if num_node_types == 1:
                if task == 'clone':
                    return {'num_nodes': num_nodes, 'last_stmts': last_stmt_indices}, ast_node_index, batch_tensor_buckets, dgl.batch(graphs), in_degrees, torch.tensor(labels)
    return fn

def make_clone_data_loader(dataset, vocab_obj, batch_size, task = 'clone', dgl_format = True, num_node_types = 1, shuffle = False, training = True, num_workers = 0, sampler = None, triplet = False):
    if sampler is not None:
        return DataLoader(dataset, batch_sampler = sampler, collate_fn = (clone_train_collate)(vocab_obj, dgl_format, num_node_types, task, triplet = triplet), num_workers = num_workers)
    
    return DataLoader(dataset, batch_size = batch_size, collate_fn = (clone_train_collate)(vocab_obj, dgl_format, num_node_types, task, triplet = True), shuffle = shuffle, num_workers = num_workers)
