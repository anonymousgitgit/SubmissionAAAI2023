import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import os
import numpy as np
import dgl
import torch
import random
from torch_geometric.data import HeteroData
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
import jsonlines
import copy
import math

from data.convert_c import convert_into_c_graph
from data.convert_cpp import convert_into_cpp_graph
from data.convert_java import convert_into_java_graph
from data.tokenizer import split_identifier_into_parts

class OJCloneDataset(Dataset):
    def __init__(self, vocab_obj, json_path, parser, dgl_format, num_node_types, triplet = False):
        self.parser = parser
        self.vocab_obj = vocab_obj
        self.dgl_format = dgl_format
        self.num_node_types = num_node_types

        self.label_examples = {}
        self.examples = []
        self.triplet = triplet
        with jsonlines.open(json_path) as f:
            for line in f:
                if line['label'] not in self.label_examples:
                    self.label_examples[line['label']] = []
                self.label_examples[line['label']].append(line)
                self.examples.append(line)
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        item = self.examples[index]
        label = item['label']
        ex = self.process_one_sample(item)
        if self.triplet is False:
            return ex, label
        
        labels = list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example=random.sample(self.label_examples[label],1)[0]
            if shuffle_example['index'] !=index:
                p_example=shuffle_example
                break
        n_example = random.sample(self.label_examples[random.sample(labels, 1)[0]], 1)[0]
        ex = self.process_one_sample(item)
        p_ex = self.process_one_sample(p_example)
        n_ex = self.process_one_sample(n_example)
        return ex, p_ex, n_ex, label
    def process_one_sample(self, item):
        text = item['code']
        label = item['label']
        refactor_json, ast_nodes, raw_stmts, stmt_indices, G = convert_into_c_graph(self.parser.parse(text.encode()), text)
        tree_data = list(map(self.convert_raw_ast_into_full_format, raw_stmts))
        ast_ids = sorted(list(set(range(G.number_of_nodes())) - set(stmt_indices)))
        ast_mapping = dict(zip(ast_ids, range(len(ast_ids))))
        stmt_mapping = dict(zip(sorted(stmt_indices), range(len(stmt_indices))))
        if self.dgl_format:
            if self.num_node_types == 1:
                return_value = ast_nodes, list(map(self.extract_tree, tree_data)), *self.convert_nx_into_dgl(G, self.num_node_types, ast_mapping, stmt_mapping)
            elif self.num_node_types == 2:
                return_value = self.adjust_ast_node_index(ast_nodes, ast_mapping), self.adjust_tree_index(list(map(self.extract_tree, tree_data)), stmt_mapping), *self.convert_nx_into_dgl(G, self.num_node_types, ast_mapping, stmt_mapping), {'ast_nodes': G.number_of_nodes() - len(stmt_indices), 'stmt_nodes': len(stmt_indices)}
        else:
            return_value = self.adjust_ast_node_index(ast_nodes, ast_mapping), self.adjust_tree_index(list(map(self.extract_tree, tree_data)), stmt_mapping), *self.convert_nx_into_pyg(G, stmt_indices, ast_mapping, stmt_mapping), {'ast_nodes': G.number_of_nodes() - len(stmt_indices), 'stmt_nodes': len(stmt_indices)}
        return return_value
    def adjust_ast_node_index(self, ast_nodes, ast_mapping):
        new_data = {}
        for k, v in ast_nodes.items():
            new_data[ast_mapping[k]] = v
        return new_data
    def adjust_tree_index(self, tree_data, stmt_mapping):
        data = []

        for tree in tree_data:
            tree['tree_index'] = stmt_mapping[tree['tree_index']]
            data.append(tree)
        return data
    def convert_raw_ast_into_full_format(self, root):
        new_root = {'node_index': root['node_index'], 'children': []}
        queue = [root]
        full_queue = [new_root]
        size = 0
        while queue:
            current_node = queue.pop(0)
            current_full_node = full_queue.pop(0)
            size += 1
            sub_tokens = split_identifier_into_parts(current_node['node_token'])
            sub_token_ids = list(map(self.vocab_obj.get_id_from_sub_token, sub_tokens))
            node_type_id = self.vocab_obj.get_id_from_node_type(current_node['node_type'])
            
            current_full_node['node_type'] = current_node['node_type']
            current_full_node['node_type_id'] = node_type_id
            current_full_node['node_token'] = current_node['node_token']
            current_full_node['node_sub_tokens'] = sub_tokens
            current_full_node['node_sub_token_ids'] = sub_token_ids
            # current_full_node['children'] = []
            
            for child in current_node['children']:
                _child = {'children': []}
                current_full_node['children'].append(_child)
                queue.append(child)
                full_queue.append(_child)
        return {'tree': new_root, 'size': size}
    def extract_tree(self, tree_data):
        tree, size = tree_data["tree"], tree_data["size"]
        # print("Extracting............", file_path)
        # print(tree)
        node_type_id = []
        node_token = []
        node_sub_tokens_id = []
        node_index = []
        node_height = []

        children_index = []
        children_node_type_id = []
        children_node_token = []
        children_node_sub_tokens_id = []
        # label = 0

        # print("Label : " + str(label))
        queue = [(tree, -1, 0)]
        # print queue
        while queue:
            # print "############"
            node, parent_ind, height = queue.pop(0)
            # print node
            # print parent_ind
            node_ind = len(node_type_id)
            # print "node ind : " + str(node_ind)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind, height + 1) for child in node['children']])
            # create a list to store this node's children indices
            children_index.append([])
            # children_node_type_id.append([])
            # children_node_token.append([])
            # children_node_sub_tokens_id.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children_index[parent_ind].append(node_ind)
                # children_node_type_id[parent_ind].append(int(node["node_type_id"]))
                # children_node_token[parent_ind].append(node["node_token"])
                # children_node_sub_tokens_id[parent_ind].append(node["node_sub_token_ids"])
            # print("a")
            # print(children_node_types)
            # print("b")
            # print(children_node_sub_tokens_id)
            node_type_id.append(node['node_type_id'])
            node_token.append(node['node_token'])
            node_sub_tokens_id.append(node['node_sub_token_ids'])
            node_index.append(node_ind)
            node_height.append(height)

        results = {}
        results['tree_index'] = tree['node_index']
        results["node_index"] = node_index
        results["node_type_id"] = node_type_id
        results["node_token"] = node_token
        results['node_height'] = node_height
        results["node_sub_token_ids"] = node_sub_tokens_id
        results["children_index"] = children_index
        # results["children_node_type_id"] = children_node_type_id
        # results["children_node_token"] = children_node_token
        # results["children_node_sub_token_ids"] = children_node_sub_tokens_id
        results["size"] = size
        return results
    def convert_nx_into_dgl(self, nx_graph, num_node_types, ast_mapping, stmt_mapping):
        
        if num_node_types == 1:
            # graph_data = defaultdict(list)
            edge_type = ['ast_edge', 'next_stmt_edge', 'control_flow_edge', 'data_flow_edge']
            graph_data = {}
            for edge_type in edge_type:
                graph_data['node', edge_type, 'node'] = []
                # graph_data['node', 'inverse_' + edge_type, 'node'] = []
            for u, v, edge_type in nx_graph.edges(data = 'edge_type'):
                graph_data[('node', edge_type, 'node')].append(torch.tensor([u, v]))
            in_degrees = []
            node_index = []
            
            for idx, in_deg in nx_graph.in_degree():
                node_index.append(idx)
                in_degrees.append(in_deg)
            in_degrees = np.array(in_degrees)[np.argsort(node_index)]
            
            node_index = []
            out_degrees = []
            for idx, out_deg in nx_graph.out_degree():
                node_index.append(idx)
                out_degrees.append(out_deg)
            out_degrees = np.array(out_degrees)[np.argsort(node_index)]
            degrees = {'node': in_degrees}, {'node': out_degrees}
        elif num_node_types == 2:
            node_types = ('ast_node', 'stmt_node')
            edge_types = ('ast_edge', 'control_flow_edge', 'next_stmt_edge')
            degrees = None, None
            all_relations = [
                ('ast_node', 'ast_edge', 'ast_node'),
                ('ast_node', 'ast_edge', 'stmt_node'),
                ('ast_node', 'control_flow_edge', 'ast_node'),
                ('ast_node', 'control_flow_edge', 'stmt_node'),
                ('stmt_node', 'ast_edge', 'ast_node'),
                ('stmt_node', 'ast_edge', 'stmt_node'),
                ('stmt_node', 'control_flow_edge', 'ast_node'),
                ('stmt_node', 'control_flow_edge', 'stmt_node'),
                ('stmt_node', 'next_stmt_edge', 'stmt_node')
            ]
            graph_data = {rel: [] for rel in all_relations}
            for u, v, k, edge_type in nx_graph.edges(data = 'edge_type', keys = True):
                s, t = list(map(int, list(k)[:-1]))
                meta_relation = node_types[s], edge_type, node_types[t]
                new_u = stmt_mapping[u] if s == 1 else ast_mapping[u]
                new_v = stmt_mapping[v] if t == 1 else ast_mapping[v]
                graph_data[meta_relation].append(torch.tensor([new_u, new_v])) 
        return dgl.heterograph(graph_data), *degrees
    def convert_nx_into_pyg(self, nx_graph, stmt_indices, ast_mapping, stmt_mapping):
       
        node_types = ('ast_node', 'stmt_node')
        edge_types = ('ast_edge', 'control_flow_edge', 'next_stmt_edge')

        all_edges = {}
        for src_type in node_types:
            for edge_type in edge_types:
                for target_type in node_types:
                    all_edges[src_type, edge_type, target_type] = []
        for u, v, k, edge_type in nx_graph.edges(data = 'edge_type', keys = True):
            s, t = list(map(int, list(k)[:-1]))
            meta_relation = node_types[s], edge_type, node_types[t]
            new_u = stmt_mapping[u] if s == 1 else ast_mapping[u]
            new_v = stmt_mapping[v] if t == 1 else ast_mapping[v]
            all_edges[meta_relation].append((new_u, new_v)) 
        data = HeteroData()
        data['stmt_node'].x = torch.Tensor(len(stmt_mapping), 1)
        data['ast_node'].x = torch.Tensor(len(ast_mapping), 1)
        for meta_relation, edges in all_edges.items():
            if len(edges) == 0: continue
            # print(torch.tensor(edges).T.shape)
            data[meta_relation].edge_index = torch.tensor(edges).T
        degrees = None, None
        return data, *degrees


class OJSampler(Sampler):
    def __init__(self, data, p, k):
        self.data = data
        self.p = p
        self.k = k
    def __iter__(self):
        data = copy.deepcopy(self.data)
        clone_data = copy.deepcopy(self.data)
        total_batch = []
        while data:
            rlbs, rp = [], []
            for k, v in data.items():
                rlbs.append(k)
                rp.append(len(v))
            rp = np.array(rp)
            rp = rp / rp.sum()
            if self.p > len(rlbs):
                rest_lbs = list(set(clone_data.keys()) - set(rlbs))
                pids = rlbs + np.random.choice(rest_lbs, self.p - len(rlbs), replace=False).tolist()
            else:
                pids = np.random.choice(rlbs, self.p, replace=False, p = rp)
            
            batch = []
            for pid in pids:
                solutions = data.get(pid, None)
                flag = True
                if solutions is None:
                    flag = False
                    solutions = clone_data[pid]
                solution_num = len(solutions)
                p_sample_num = min(solution_num, self.k)
                if solution_num < self.k:
                    _tmp = clone_data[pid]
                    random.shuffle(_tmp)
                    _tmp = list(set(_tmp) - set(solutions))
                    _tmp = solutions + _tmp[:self.k - solution_num]
                    batch.extend(_tmp)
                else:  
                    random.shuffle(solutions)
                    batch.extend(solutions[:p_sample_num])
                if flag:
                    data[pid] = solutions[p_sample_num:]
                if pid in data and len(data[pid]) == 0:
                    data.pop(pid)
            total_batch.append(batch)
        random.shuffle(total_batch)
        return iter(total_batch)
    def __len__(self):
        length = sum([len(list(v)) for v in self.data.values()])
        return int(math.ceil(length / (self.k * self.p)))

class OJXGLueSampler(Sampler):
    def __init__(self, data, p = 8, k = 1):
        self.data = data
        self.p = p
        if type(k) is int:
            self.k = [k] * p
        elif type(k) is list:
            self.k = k
    def __iter__(self):
        data = copy.deepcopy(self.data)
        clone_data = copy.deepcopy(self.data)
        total_batch = []
        while data:
            rlbs, rp = [], []
            for k, v in data.items():
                rlbs.append(k)
                rp.append(len(v))
            rp = np.array(rp)
            rp = rp / rp.sum()
            if self.p > len(rlbs):
                rest_lbs = list(set(clone_data.keys()) - set(rlbs))
                pids = rlbs + np.random.choice(rest_lbs, self.p - len(rlbs), replace=False).tolist()
            else:
                pids = np.random.choice(rlbs, self.p, replace=False, p = rp)
            
            batch = []
            for pid, k_c in zip(pids, self.k):
                solutions = data.get(pid, None)
                flag = True
                if solutions is None:
                    flag = False
                    solutions = clone_data[pid]
                solution_num = len(solutions)
                p_sample_num = min(solution_num, k_c)
                if solution_num < k_c:
                    _tmp = clone_data[pid]
                    random.shuffle(_tmp)
                    _tmp = list(set(_tmp) - set(solutions))
                    _tmp = solutions + _tmp[:k_c - solution_num]
                    batch.extend(_tmp)
                else:  
                    random.shuffle(solutions)
                    batch.extend(solutions[:p_sample_num])
                if flag:
                    data[pid] = solutions[p_sample_num:]
                if pid in data and len(data[pid]) == 0:
                    data.pop(pid)
            total_batch.append(batch)
        random.shuffle(total_batch)
        return iter(total_batch)
    def set_value(self, p, k):
        self.p = p
        if type(k) is int:
            self.k = [k] * p
        elif type(k) is list:
            self.k = k
    def __len__(self):
        length = sum([len(list(v)) for v in self.data.values()])
        bs = sum(self.k)
        return int(math.ceil(length / bs))
            