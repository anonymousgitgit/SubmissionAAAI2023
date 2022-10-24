import numpy as np
from constants.vocab import PAD_IDX
import torch
import torch.nn as nn
import random
import dgl
# from torch_geometric.data import Batch
import random

from network.graph_layers import *
from network.decoder import DecoderRNN, Embedder, AttnDecoderRNN
from network.layers import ArcMarginProduct
from data.utils import create_target_mask

class ModelFromCp(nn.Module):
    def __init__(self, pretrained_model, config, dgl_format, num_node_types, *, graph_aggr):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.dgl_format = dgl_format
        self.graph_aggr = graph_aggr
        self.num_node_types = num_node_types
        self.fc_layer = nn.Sequential(
            # nn.Linear(num_node_types * config['out_channels'], num_node_types * config['out_channels']),
            # nn.Dropout(0.3),
            # nn.ReLU(),
            nn.Linear(num_node_types * config['out_channels'], num_node_types * config['out_channels']),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(num_node_types * config['out_channels'], config['n_classes'])
        )
        self.attn_graph_layer = AttentionGraphLevel(config['out_channels'], graph_aggr)
    def forward(self, num_nodes, ast_node_index, buckets, graphs, in_degrees):
        ast_node_index, ast_node_embeddings = self.pretrained_model.ast_embedding_layer(ast_node_index)
        # print('ast node', np.array(ast_node_index).shape, np.unique(ast_node_index).shape)
        sizes = list(buckets.keys())
        if self.training:
            random.shuffle(sizes)
        node_embeddings, batch_tree_index = [], []
        for size in sizes:
            batch = buckets[size]
            each_bucket_embeddings = self.pretrained_model.tbcnn_layer(batch['batch_node_index'], batch['batch_node_type_id'], batch['batch_node_height'], batch['batch_node_sub_tokens_id'], batch['batch_children_index'])
            node_embeddings.append(each_bucket_embeddings)
            batch_tree_index.extend(batch['batch_tree_index'])
        # print('both:', graphs.num_nodes(), len(ast_node_index) + len(batch_tree_index), len(set(ast_node_index) & set(batch_tree_index)), len(set(ast_node_index) | set(batch_tree_index)))
        # print('stmt node', np.array(batch_tree_index).shape, np.unique(batch_tree_index).shape)
        stmt_node_embeddings = torch.cat(node_embeddings, axis = 0)
        if self.dgl_format:
            if self.num_node_types == 1:
                embeddings = torch.cat((ast_node_embeddings, stmt_node_embeddings), dim = 0)
                # ensure that after merging, the order of nodes matches the order in graph
                order_indices = np.argsort(ast_node_index + batch_tree_index)
                # print('a', order_indices.shape)
                node_embeddings = embeddings[order_indices]
                data = {
                    'node': node_embeddings
                }
            else:
                ast_order_node_embeddings = ast_node_embeddings[np.argsort(ast_node_index)]
                stmt_order_node_embeddings = stmt_node_embeddings[np.argsort(batch_tree_index)]
                data = {
                    'ast_node': ast_order_node_embeddings,
                    'stmt_node': stmt_order_node_embeddings
                }
            all_node_embeddings = self.pretrained_model.hgt_graph_layer(data, graphs = graphs)
        else:
            ast_order_node_embeddings = ast_node_embeddings[np.argsort(ast_node_index)]
            stmt_order_node_embeddings = stmt_node_embeddings[np.argsort(batch_tree_index)]
            graphs['ast_node'].x = ast_order_node_embeddings
            graphs['stmt_node'].x = stmt_order_node_embeddings
            # print(graphs.x_dict)
            all_node_embeddings = self.pretrained_model.hgt_graph_layer(graphs.x_dict, edge_index_dict = graphs.edge_index_dict)

        graph_embeddings = self.attn_graph_layer(all_node_embeddings['node'], num_nodes['num_nodes'], all_node_embeddings['node'][num_nodes['last_stmts']]) 
        logits = self.fc_layer(graph_embeddings)
        return graph_embeddings, logits

    def get_groups(self, num_nodes):
        groups = []
        for index, num_node in enumerate(num_nodes):
            groups.extend([index] * num_node)
        return torch.tensor(groups)

class TripletModelFromCp(nn.Module):
    def __init__(self, pretrained_model, config, metadata, dgl_format, func, num_node_types, *, graph_aggr):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.dgl_format = dgl_format
        self.graph_aggr = graph_aggr
        self.num_node_types = num_node_types
        self.mlp = nn.Sequential(
            nn.Linear(config['out_channels'], config['out_channels'] // 2),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(config['out_channels'] // 2, config['out_channels'] // 2),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(config['out_channels'] // 2, num_node_types * config['out_channels'] // 2),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2)            
        )
        self.fc_layer = nn.Linear(num_node_types * config['out_channels'] // 2, config['n_classes'])
        self.attn_graph_layer = AttentionGraphLevel(config['out_channels'], graph_aggr)
    def forward(self, num_nodes, ast_node_index, buckets, graphs, in_degrees):
        ast_node_index, ast_node_embeddings = self.pretrained_model.ast_embedding_layer(ast_node_index)
        # print('ast node', np.array(ast_node_index).shape, np.unique(ast_node_index).shape)
        sizes = list(buckets.keys())
        if self.training:
            random.shuffle(sizes)
        node_embeddings, batch_tree_index = [], []
        for size in sizes:
            batch = buckets[size]
            each_bucket_embeddings = self.pretrained_model.tbcnn_layer(batch['batch_node_index'], batch['batch_node_type_id'], batch['batch_node_height'], batch['batch_node_sub_tokens_id'], batch['batch_children_index'])
            node_embeddings.append(each_bucket_embeddings)
            batch_tree_index.extend(batch['batch_tree_index'])
        # print('both:', graphs.num_nodes(), len(ast_node_index) + len(batch_tree_index), len(set(ast_node_index) & set(batch_tree_index)), len(set(ast_node_index) | set(batch_tree_index)))
        # print('stmt node', np.array(batch_tree_index).shape, np.unique(batch_tree_index).shape)
        stmt_node_embeddings = torch.cat(node_embeddings, axis = 0)
        if self.dgl_format:
            if self.num_node_types == 1:
                embeddings = torch.cat((ast_node_embeddings, stmt_node_embeddings), dim = 0)
                # ensure that after merging, the order of nodes matches the order in graph
                order_indices = np.argsort(ast_node_index + batch_tree_index)
                # print('a', order_indices.shape)
                node_embeddings = embeddings[order_indices]
                data = {
                    'node': node_embeddings
                }
            else:
                ast_order_node_embeddings = ast_node_embeddings[np.argsort(ast_node_index)]
                stmt_order_node_embeddings = stmt_node_embeddings[np.argsort(batch_tree_index)]
                data = {
                    'ast_node': ast_order_node_embeddings,
                    'stmt_node': stmt_order_node_embeddings
                }
            all_node_embeddings = self.pretrained_model.hgt_graph_layer(data, graphs = graphs)
        else:
            ast_order_node_embeddings = ast_node_embeddings[np.argsort(ast_node_index)]
            stmt_order_node_embeddings = stmt_node_embeddings[np.argsort(batch_tree_index)]
            graphs['ast_node'].x = ast_order_node_embeddings
            graphs['stmt_node'].x = stmt_order_node_embeddings
            # print(graphs.x_dict)
            all_node_embeddings = self.pretrained_model.hgt_graph_layer(graphs.x_dict, edge_index_dict = graphs.edge_index_dict)

        graph_embeddings = self.attn_graph_layer(all_node_embeddings['node'], num_nodes['num_nodes'], all_node_embeddings['node'][num_nodes['last_stmts']]) 
        embeds = self.mlp(graph_embeddings)
        logits = self.fc_layer(embeds)
        return embeds, logits

    def get_groups(self, num_nodes):
        groups = []
        for index, num_node in enumerate(num_nodes):
            groups.extend([index] * num_node)
        return torch.tensor(groups)
