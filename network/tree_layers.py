import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from torch_geometric.nn import HGTConv
from torch_scatter import scatter_mean, scatter_add, scatter_max
import math

from network.ops import gather_nd, group

class PositionalEmbedding(nn.Module):
    def __init__(self, 
                channels,
                max_index):
        super().__init__()
        position = torch.arange(max_index).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * -math.log(1e4) / channels)
        emb = nn.Embedding(max_index + 1, channels, padding_idx = 0)
        emb.weight.data[1:, ::2] = torch.sin(position * div_term) / math.sqrt(channels)
        emb.weight.data[1:, 1::2] = torch.cos(position * div_term) / math.sqrt(channels)
        emb.requires_grad = False
        self.emb = emb
        self.linear = nn.Linear(channels, channels)
    def forward(self, x, t):
        return x + self.linear(self.emb(t.long()))

class NodeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config['vocab_token_size'] + 1, config['in_channels'] // 2, padding_idx = 0)
        self.type_embedding = nn.Embedding(config['vocab_type_size'] + 1, config['in_channels'] // 2, padding_idx = 0)
    def forward(self, type_index, sub_token_ids, reduce_dim, concat_dim):
        type_embedding = self.get_type_embedding(type_index)
        token_embedding = self.get_token_embedding(sub_token_ids)
        if reduce_dim is not None:
            token_embedding = token_embedding.sum(dim = reduce_dim)
        return torch.cat((type_embedding, token_embedding), dim = concat_dim)
    def get_token_embedding(self, indices):
        token_embedding = self.token_embedding(indices.long())
        return token_embedding
    def get_type_embedding(self, indices):
        type_embedding = self.type_embedding(indices.long())
        return type_embedding
class ASTNodeEmbedding(nn.Module):
    def __init__(self, embedding_layer, tree_lstm):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.tree_lstm = tree_lstm

    def forward(self, ast_nodes):
        ast_node_index, ast_node_embeddings = [], []
        if self.tree_lstm:
            for node, info in ast_nodes.items():
                node_embedding = self.embedding_layer(info['node_type_index'], info['node_sub_token_ids'], reduce_dim = 0, concat_dim = -1)
                ast_node_index.append(node)
                ast_node_embeddings.append(node_embedding)
            ast_node_embeddings = torch.stack(ast_node_embeddings, dim = 0)
        else:
            for single_ast_nodes in ast_nodes:
                ast_ids = []
                ast_embeddings = []
                for node, info in single_ast_nodes.items():
                    node_embedding = self.embedding_layer(info['node_type_index'], info['node_sub_token_ids'], reduce_dim = 0, concat_dim = -1)
                    ast_ids.append(node)
                    ast_embeddings.append(node_embedding)
                ast_node_index.append(ast_ids)
                ast_embeddings = torch.stack(ast_embeddings, dim = 0)
                ast_node_embeddings.append(ast_embeddings)
        # ast_node_embeddings = self.mlp(ast_node_embeddings)
        return ast_node_index, ast_node_embeddings
            
class GRUStmtLayer(nn.Module):
    def __init__(self, embedding_layer, in_channels, num_layers: int = 1):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.in_channels = in_channels
        self.gru = nn.GRU(in_channels, in_channels // 2, num_layers = num_layers, batch_first = True, bidirectional = True)
    def forward(self, stmt_type_ids, stmt_token_ids, stmt_lengths):
        embs = self.embedding_layer(stmt_type_ids, stmt_token_ids, reduce_dim = 2, concat_dim = 2)
        embs = nn.utils.rnn.pack_padded_sequence(embs, stmt_lengths, batch_first = True, enforce_sorted = False)
        return self.gru(embs)[1].transpose(0, 1).reshape(-1, self.in_channels)

class TBCNNLayer(nn.Module):
    def __init__(self, embedding_layer: nn.Module, config: dict, aggr: str = 'attention'):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.aggr = aggr
        self.use_norm = config['normalization'] is not None
        
        assert self.in_channels % 2 == 0, "the number of in channels must be even"

        self.attn_w = nn.Parameter(torch.randn(self.out_channels, 1))
        self.w_t_lst = nn.ParameterList([nn.Parameter(torch.randn(self.in_channels, self.out_channels)) for _ in range(self.num_layers)])
        self.w_l_lst = nn.ParameterList([nn.Parameter(torch.randn(self.in_channels, self.out_channels)) for _ in range(self.num_layers)])
        self.w_r_lst = nn.ParameterList([nn.Parameter(torch.randn(self.in_channels, self.out_channels)) for _ in range(self.num_layers)])
        self.bias_lst = nn.ParameterList([nn.Parameter(torch.zeros(self.out_channels, )) for _ in range(self.num_layers)])
        self.leaky_relu_lst = nn.ModuleList([nn.LeakyReLU(inplace = True) for _ in range(self.num_layers)])
        self.dropout_lst = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(self.num_layers)])
    
        self.norm_layers = nn.ModuleList([nn.LayerNorm(self.out_channels) for _ in range(self.num_layers)])

        if self.aggr == 'attention':
            self.depth_embedding = nn.Embedding(8, 1, padding_idx = 0)        
            self.query_trans = nn.Linear(self.out_channels, self.out_channels)
            self.key_trans = nn.Linear(self.out_channels, self.out_channels)
            self.value_trans = nn.Linear(self.out_channels, self.out_channels)
            # self.control_gate = nn.Linear(2 * self.out_channels, 1)
            self.gate = nn.Parameter(torch.zeros(1))

        self.init_params()
    def init_params(self):
        for w_t, w_l, w_r in zip(self.w_t_lst, self.w_l_lst, self.w_r_lst):
            nn.init.xavier_normal_(w_t)
            nn.init.xavier_normal_(w_l)
            nn.init.xavier_normal_(w_r)
        nn.init.xavier_normal_(self.attn_w)
    def forward(self, parent_node_embedding, children_index):
        # parent_node_type_embedding = self.get_parent_type_embedding(node_type_index)
        # parent_node_token_embedding = self.get_parent_token_embedding(node_token_ids)

        # parent_node_embedding = torch.cat((parent_node_type_embedding, parent_node_token_embedding), dim = 2)  
        # children_type_embedding = self.get_children_embedding_from_parent(parent_node_type_embedding, children_index)
        # children_token_embedding = self.get_children_tokens_embedding(children_node_token_ids)
        children_embedding = self.get_children_embedding_from_parent(parent_node_embedding, children_index)
        # children_embedding = torch.cat((children_type_embedding, children_token_embedding), dim = 3)

        # parent_node_embedding = self.parent_mlp(parent_node_embedding)
        # children_embedding = self.children_mlp(children_embedding)
        node_embedding = self.conv_op(parent_node_embedding, children_embedding, children_index)

        return self.aggregate(node_embedding)
    def aggregate(self, node_embedding):
        # return torch.max(node_embedding, 1)[0]
        if self.aggr == 'attention':
            # scores = torch.matmul(node_embedding, self.attn_w)
            # node_type_index = node_type_index.unsqueeze(-1)
            # scores = torch.where(node_type_index == -1, torch.tensor(float('-inf'), device = node_embedding.device), scores)
            # attn_coefficients = F.softmax(scores, dim = 1)
            
            # tree_embeddings =  torch.sum(node_embedding * attn_coefficients, dim = 1)
            children_embedding = node_embedding[:, 1:]
            root_embedding = node_embedding[:, 0]

            key_embedding = self.key_trans(children_embedding)
            query_embedding = self.query_trans(root_embedding)
            value_embedding = self.value_trans(children_embedding)
            logits = torch.einsum('ij, ikj -> ik', query_embedding, key_embedding)
            scores = F.softmax(logits, dim = 1)
            agg_children_embedding = torch.einsum('ik, ikj -> ij', scores, value_embedding)

            # control_gate = torch.sigmoid(self.control_gate(torch.cat((root_embedding, agg_children_embedding), dim = 1)))
            control_gate = torch.sigmoid(self.gate)
            tree_embeddings = control_gate * root_embedding + (1 - control_gate) * agg_children_embedding
        elif self.aggr == 'max-pooling':
            tree_embeddings = torch.max(node_embedding, dim = 1)[0]
        return tree_embeddings

    def conv_op(self, parent_node_embedding, children_embedding, children_index):
        outputs = [parent_node_embedding]
        for i in range(self.num_layers):
            parent_node_embedding = self.conv_step(torch.sum(torch.stack(outputs, 0), 0), children_embedding, children_index, w_t = self.w_t_lst[i], w_l = self.w_l_lst[i], w_r = self.w_r_lst[i], bias = self.bias_lst[i])
            if self.use_norm:
                parent_node_embedding = self.norm_layers[i](parent_node_embedding)
            parent_node_embedding = self.leaky_relu_lst[i](parent_node_embedding)
            parent_node_embedding = self.dropout_lst[i](parent_node_embedding)
            children_embedding = self.get_children_embedding_from_parent(parent_node_embedding, children_index)
            outputs.append(parent_node_embedding)
        return parent_node_embedding
    def conv_step(self, parent_node_embedding, children_embedding, children_index, w_t, w_l, w_r, bias):
        parent_node_embedding = parent_node_embedding.unsqueeze(2)
        tree_embedding = parent_node_embedding if np.prod(children_embedding.shape) == 0 else torch.cat((parent_node_embedding, children_embedding), dim = 2)
        eta_t = self.eta_t(children_index)
        eta_r = self.eta_r(children_index, eta_t)
        eta_l = self.eta_l(children_index, eta_t, eta_r)
        eta = torch.stack((eta_t, eta_l, eta_r), dim = -1)
        weights = torch.stack((w_t, w_l, w_r), dim = 0)
        result = torch.matmul(tree_embedding.permute(0, 1, 3, 2), eta)
        result = torch.tensordot(result, weights, dims = ([3, 2], [0, 1]))
        return result + bias

    def get_children_embedding_from_parent(self, parent_node_embedding, children_index):
        batch_size, num_nodes, num_children = children_index.shape
        channels = parent_node_embedding.shape[-1]
        zero_vecs = torch.zeros(batch_size, 1, channels).to(parent_node_embedding.device)
        lookup_table = torch.cat((zero_vecs, parent_node_embedding[:, 1:]), dim = 1)
        children_index = children_index.unsqueeze(-1)
        batch_index = torch.arange(batch_size).view(batch_size, 1, 1, 1).to(parent_node_embedding.device)
        batch_index = torch.tile(batch_index, (1, num_nodes, num_children, 1))
        children_index = torch.cat((batch_index, children_index), dim = -1)
        return gather_nd(lookup_table, children_index)

    def get_parent_type_embedding(self, parent_node_type_index):
        return self.embedding_layer.get_type_embedding(parent_node_type_index, True)
    def get_parent_token_embedding(self, parent_node_token_ids):
        return self.embedding_layer.get_token_embedding(parent_node_token_ids, True).sum(dim = 2)
    def get_children_tokens_embedding(self, children_node_token_ids):
        return self.embedding_layer.get_token_embedding(children_node_token_ids, True).sum(dim = 3)
    

    def eta_t(self, children):
        batch_size, num_nodes, num_children = children.shape
        return torch.tile(torch.unsqueeze(
            torch.cat((torch.ones(num_nodes, 1), torch.zeros(num_nodes, num_children)), dim = 1),
            dim = 0
        ), (batch_size, 1, 1)).to(children.device)
    def eta_r(self, children, eta_t):
        batch_size, num_nodes, num_children = children.shape
        if num_children == 0:
            return torch.zeros(batch_size, num_nodes, 1).to(children.device)
        num_siblings = torch.tile(torch.count_nonzero(children, dim = 2).unsqueeze(-1), (1, 1, num_children + 1))
        mask = torch.cat((torch.zeros(batch_size, num_nodes, 1).to(children.device), torch.minimum(children, torch.ones_like(children).to(children.device))), dim = 2).to(children.device)
        child_indices = torch.tile(torch.arange(-1, num_children).unsqueeze(0).unsqueeze(0), (batch_size, num_nodes, 1)).to(children.device) * mask
        singles = torch.cat((
            torch.zeros(batch_size, num_nodes, 1),
            torch.full((batch_size, num_nodes, 1), 0.5),
            torch.zeros(batch_size, num_nodes, num_children - 1)
        ), dim = 2).to(children.device)
        return torch.where(num_siblings == 1, singles, (1 - eta_t) * child_indices / (num_siblings - 1))
    def eta_l(self, children, eta_t, eta_r):
        batch_size, num_nodes, _ = children.shape
        mask = torch.cat((torch.zeros(batch_size, num_nodes, 1, device = children.device), torch.minimum(children, torch.ones_like(children, device = children.device))), dim = 2)
        return mask * (1 - eta_t) * (1 - eta_r)


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_iou = nn.Linear(input_size, 3 * hidden_size, bias = False)
        self.U_iou = nn.Linear(input_size, 3 * hidden_size, bias = False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * hidden_size))
        self.U_f = nn.Linear(hidden_size, hidden_size)
    def message_func(self, edges):
        return {
            'h': edges.src['h'],
            'c': edges.src['c']
        }
    def reduce_func(self, nodes):
        h_tile = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {
            'iou': nodes.data['iou'] + self.U_iou(h_tile),
            'c': c
        }
    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {
            'h': h,
            'c': c
        }

class TreeAggregation(nn.Module):
    def forward(self, embeddings, tree_sizes):
        groups = self.get_groups(tree_sizes).to(embeddings.device)
        tree_embeddings = scatter_max(embeddings, groups, dim = 0)[0]
        return tree_embeddings
    def get_groups(self, num_nodes):
        groups = []
        for index, num_node in enumerate(num_nodes):
            groups.extend([index] * num_node)
        return torch.tensor(groups)

if __name__ == "__main__":
    TBCNNLayer(10, 10, 3)
        