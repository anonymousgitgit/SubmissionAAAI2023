import torch
import torch.nn as nn
import random
import numpy as np

from network.graph_layers import *

class CloneASTNNModel(nn.Module):   
    def __init__(self, pretrained_model, out_channels, dgl_format, num_node_types, graph_aggr):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.projector = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels),
            # nn.ReLU(inplace = True),
            # nn.Dropout(0.15),
            # nn.Linear(256, 256)
            # nn.Identity()
        )
        self.attn_graph_layer = AttentionGraphLevel(out_channels, graph_aggr)
        self.dgl_format = dgl_format
        self.num_node_types = num_node_types
        self.temperature = 0.5
        self.tau_plus = 0.1 
        self.debiased = True

        self.hidden2label = nn.Linear(out_channels, 1)
    
        
    def forward(self, num_nodes, ast_node_index, buckets, graphs, in_degrees, labels): 
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
        # embeddings = self.projector(graph_embeddings[:, :256])
        embeddings = self.projector(graph_embeddings)
        # return embeddings
        batch_size = embeddings.shape[0] // 2
        
        vec1, vec2 = embeddings.split(batch_size, 0)
        abs_dict = torch.abs(vec1 - vec2)
        y = torch.sigmoid(self.hidden2label(abs_dict))
        return y.squeeze(1)