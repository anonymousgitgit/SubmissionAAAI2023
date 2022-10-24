import numpy as np
import torch
import torch.nn as nn
import dgl


from network.tree_layers import NodeEmbedding, GRUStmtLayer, ASTNodeEmbedding, TBCNNLayer, ChildSumTreeLSTMCell, TreeAggregation

class GRUTBCNN(nn.Module):
    def __init__(self, config, tree_aggr):
        super().__init__()
        self.node_embedding_layer = NodeEmbedding(config)
        self.ast_embedding_layer = ASTNodeEmbedding(self.node_embedding_layer, False)
        self.gru_stmt_layer = GRUStmtLayer(self.node_embedding_layer, config['out_channels'])
        self.tbcnn_layer = TBCNNLayer(self.node_embedding_layer, config, tree_aggr)
        self.fc_layer = nn.Linear(config['out_channels'], config['n_classes'])
    def forward(self, ast_nodes, pad_stmt_type_ids, pad_stmt_token_ids, stmt_lengths, stmt_indices, batch_tree):
        ast_node_ids, ast_node_embeddings = self.ast_embedding_layer(ast_nodes)
        stmt_embeddings = self.gru_stmt_layer(pad_stmt_type_ids, pad_stmt_token_ids, stmt_lengths)
        offset = 0
        batch = []
        for i, (ast_ids, st_ids) in enumerate(zip(ast_node_ids, stmt_indices)):
            length = len(st_ids)
            ast_embeddings = ast_node_embeddings[i]
            
            st_embeddings = stmt_embeddings[offset: offset + length]
            offset += length
            embeds = torch.cat((ast_embeddings, st_embeddings))
            embeds = embeds[np.argsort(ast_ids + st_ids)]
            batch.append(embeds)
        pad_batch = nn.utils.rnn.pad_sequence(batch, batch_first = True)
        tree_embeddings = self.tbcnn_layer(pad_batch, batch_tree['batch_children_index'])
        logits = self.fc_layer(tree_embeddings)
        return tree_embeddings, logits

class GRUTreeLSTM(nn.Module):
    def __init__(self, config, tree_aggr):
        super().__init__()
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.node_embedding_layer = NodeEmbedding(config)
        self.ast_embedding_layer = ASTNodeEmbedding(self.node_embedding_layer, True)
        self.gru_stmt_layer = GRUStmtLayer(self.node_embedding_layer, config['out_channels'])
        self.tree_lstm_cells = nn.ModuleList([
            ChildSumTreeLSTMCell(config['in_channels'], config['out_channels'])
            for _ in range(config['num_layers'])
        ])
        self.fc_layer = nn.Linear(config['out_channels'], config['n_classes'])
        self.tree_aggregation = TreeAggregation()
    def forward(self, ast_nodes, pad_stmt_type_ids, pad_stmt_token_ids, stmt_lengths, stmt_indices, batch_tree, tree_sizes):
        ast_node_ids, ast_node_embeddings = self.ast_embedding_layer(ast_nodes)
        stmt_embeddings = self.gru_stmt_layer(pad_stmt_type_ids, pad_stmt_token_ids, stmt_lengths)
        all_embeddings = torch.cat((ast_node_embeddings, stmt_embeddings), dim = 0)
        reorder_indices = np.argsort(ast_node_ids + stmt_indices)
        all_embeddings = all_embeddings[reorder_indices]
        
        h = torch.zeros(batch_tree.number_of_nodes(), self.out_channels, device = all_embeddings.device)
        c = torch.zeros(batch_tree.number_of_nodes(), self.out_channels, device = all_embeddings.device)

        batch_tree.ndata['h'] = h
        batch_tree.ndata['c'] = c
        for i, tree_lstm_cell in enumerate(self.tree_lstm_cells):
            batch_tree.ndata['iou'] = tree_lstm_cell.W_iou(all_embeddings)
            dgl.prop_nodes_topo(batch_tree, message_func = tree_lstm_cell.message_func, 
                                reduce_func = tree_lstm_cell.reduce_func,
                                apply_node_func = tree_lstm_cell.apply_node_func)
        h = batch_tree.ndata.pop('h')
        tree_embeddings = self.tree_aggregation(h, tree_sizes)
        logits = self.fc_layer(tree_embeddings)
        return tree_embeddings, logits








