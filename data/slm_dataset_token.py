import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import os
import numpy as np
import json
import dgl
import torch
import re
import jsonlines
from torch.utils.data import Dataset

from data.mask_convert_java import convert_into_java_graph
from data.tokenizer import split_identifier_into_parts
from data.dynamic_vocab import DynamicVocab

class MaskedSLMDataset(Dataset):
    def __init__(self, vocab_obj, lang_obj, path, parser, dgl_format, num_node_types, max_seq_length, tokenizer):
        self.path = path
        self.parser = parser
        self.vocab_obj = vocab_obj
        self.lang_obj = lang_obj
        self.dgl_format = dgl_format
        self.num_node_types = num_node_types
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

        data = []
        with jsonlines.open(path) as f:
            for line in f:
                data.append(line)
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):

        text = 'public class Main{\n' + self.data[index] + '\n}'
        # text = re.sub(r'"[\s\S]*?"', ' "___str___" ', text)
        # print('path', path)
        refactor_json, ast_nodes, raw_stmts, stmt_indices, G, masked_stmt, src_tokens = convert_into_java_graph(self.parser.parse(text.encode()), text, self.max_seq_length, self.tokenizer)
        tgt_tokens = masked_stmt['tokens']
        
        tgt_tokens = ['[SOS]'] + tgt_tokens + ['[EOS]']
        tgt_token_ids = self.get_ids_from_tokens(tgt_tokens)
        src_token_ids = list(map(lambda x: self.lang_obj.get_word_index(x), src_tokens))
        src_vocab = DynamicVocab(no_special_token = True)
        src_vocab.add_tokens(src_tokens)
        extra_src_token_ids = list(map(lambda x: src_vocab[x], src_tokens))
        src_token_ids_vocab = list(map(lambda x: self.lang_obj.get_word_index(x), src_tokens))
        extra_src_info = {'src_vocab': src_vocab, 'extra_src_token_ids': extra_src_token_ids, 'src_token_ids_vocab': src_token_ids_vocab}
        
        tree_data = list(map(self.convert_raw_ast_into_full_format, raw_stmts))
        ast_ids = sorted(list(set(range(G.number_of_nodes())) - set(stmt_indices)))
        ast_mapping = dict(zip(ast_ids, range(len(ast_ids))))
        stmt_mapping = dict(zip(sorted(stmt_indices), range(len(stmt_indices))))
        if self.dgl_format:
            if self.num_node_types == 1:
                return_value = ast_nodes, list(map(self.extract_tree, tree_data)), *self.convert_nx_into_dgl(G, self.num_node_types, ast_mapping, stmt_mapping), stmt_indices, extra_src_info, src_token_ids, tgt_token_ids, tgt_tokens, [self.tokenizer.decode(tgt_token_ids[1:-1])]
            elif self.num_node_types == 2:
                return_value = self.adjust_ast_node_index(ast_nodes, ast_mapping), self.adjust_tree_index(list(map(self.extract_tree, tree_data)), stmt_mapping), *self.convert_nx_into_dgl(G, self.num_node_types, ast_mapping, stmt_mapping), {'ast_nodes': G.number_of_nodes() - len(stmt_indices), 'stmt_nodes': len(stmt_indices)}, stmt_indices, extra_src_info, src_token_ids, tgt_token_ids, tgt_tokens, [self.tokenizer.decode(tgt_token_ids[1:-1])]
        else:
            return_value = self.adjust_ast_node_index(ast_nodes, ast_mapping), self.adjust_tree_index(list(map(self.extract_tree, tree_data)), stmt_mapping), *self.convert_nx_into_pyg(G, stmt_indices, ast_mapping, stmt_mapping), {'ast_nodes': G.number_of_nodes() - len(stmt_indices), 'stmt_nodes': len(stmt_indices)}, stmt_indices, src_token_ids, tgt_token_ids, tgt_tokens, [self.tokenizer.decode(tgt_token_ids[1:-1])]
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
            # sub_tokens = split_identifier_into_parts(current_node['node_token'])
            if current_node['node_type'] == 'sub_token':
                sub_tokens = [current_node['node_token']]
            elif current_node['node_type'] == 'string_literal':
                sub_tokens = self.tokenizer.encode(' "str" ').tokens
            else:
                sub_tokens = self.tokenizer.encode(current_node['node_token']).tokens
            sub_token_ids = list(map(self.vocab_obj.get_id_from_sub_token, sub_tokens))
            node_type_id = self.vocab_obj.get_id_from_node_type(current_node['node_type'])
            
            current_full_node['node_type'] = current_node['node_type']
            current_full_node['node_type_id'] = node_type_id
            
            children = current_node['children']
            # current_full_node['node_sub_tokens'] = sub_tokens
            # current_full_node['node_sub_token_ids'] = sub_token_ids
            # current_full_node['children'] = []
            if current_full_node['node_type'] in ['identifier', 'type_identifier']:
                _sub_tokens = ['<hole>']
                current_full_node['node_sub_tokens'] = _sub_tokens
                current_full_node['node_sub_token_ids'] = list(map(self.vocab_obj.get_id_from_sub_token, _sub_tokens))
                current_full_node['node_token'] = '<hole>'
                _children = []
                for sub_token in sub_tokens:
                    _children.append(
                        {
                            'node_type': 'sub_token',
                            'node_token': sub_token,
                            'children': []
                        }
                    )
                children.extend(_children)
            else:
                current_full_node['node_token'] = current_node['node_token']
                current_full_node['node_sub_tokens'] = sub_tokens
                current_full_node['node_sub_token_ids'] = sub_token_ids
            
            for child in children:
                _child = {'children': []}
                current_full_node['children'].append(_child)
                queue.append(child)
                full_queue.append(_child)
        return {'tree': new_root, 'size': size}
    def extract_tree(self, tree_data):
        tree, size = tree_data["tree"], tree_data["size"]
        # print("Extracting............", file_path)
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
        token_node_ids = []
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
            if node['node_token'] not in ['', '<hole>']:
                token_node_ids.append(node_ind)
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
        results['token_node_ids'] = token_node_ids
        results['token_code_lens'] = len(token_node_ids)
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
            max_i = float('-inf')
            for u, v, edge_type in nx_graph.edges(data = 'edge_type'):
                graph_data[('node', edge_type, 'node')].append(torch.tensor([u, v]))
                max_i = max(max(max_i, u), v)
            node_ids = nx_graph.nodes
            # print('NODE', node_ids, max(node_ids))
            # print('max index', max_i)
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
    def get_ids_from_tokens(self, tokens):
        return list(map(lambda x: self.lang_obj.get_word_index(x), tokens))