import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
# from torch_geometric.nn import HGTConv
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

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        out = self.mlp(x)
        return out + x

# class HGTGRUConv(HGTConv):
#     def __init__(self,
#         in_channels,
#         out_channels,
#         metadata,
#         timestep,
#         heads: int = 1,
#         group: str = "sum",
#         **kwargs):
#         super().__init__(in_channels, out_channels, metadata, heads, group, **kwargs)
#         self.gru_layer = nn.GRUCell(out_channels, out_channels)
#         self.timestep = timestep
#         print('HGT GRU PYTORCH GEOMETRIC')
#     def forward(
#         self,
#         x_dict,
#         edge_index_dict
#     ):
#         r"""
#         Args:
#             x_dict (Dict[str, Tensor]): A dictionary holding input node
#                 features  for each individual node type.
#             edge_index_dict: (Dict[str, Union[Tensor, SparseTensor]]): A
#                 dictionary holding graph connectivity information for each
#                 individual edge type, either as a :obj:`torch.LongTensor` of
#                 shape :obj:`[2, num_edges]` or a
#                 :obj:`torch_sparse.SparseTensor`.

#         :rtype: :obj:`Dict[str, Optional[Tensor]]` - The ouput node embeddings
#             for each node type.
#             In case a node type does not receive any message, its output will
#             be set to :obj:`None`.
#         """

#         H, D = self.heads, self.out_channels // self.heads

#         for _ in range(self.timestep):
#             k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

#             # Iterate over node-types:
#             for node_type, x in x_dict.items():
#                 k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
#                 q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
#                 v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)
#                 out_dict[node_type] = []

#             # Iterate over edge-types:
#             for edge_type, edge_index in edge_index_dict.items():
#                 src_type, _, dst_type = edge_type
#                 edge_type = '__'.join(edge_type)

#                 a_rel = self.a_rel[edge_type]
#                 k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)

#                 m_rel = self.m_rel[edge_type]
#                 v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)

#                 # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
#                 out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
#                                     rel=self.p_rel[edge_type], size=None)
#                 out_dict[dst_type].append(out)

#             # Iterate over node-types:
#             for node_type, outs in out_dict.items():
#                 out = group(outs, self.group)

#                 if out is None:
#                     out_dict[node_type] = None
#                     continue

#                 out = self.a_lin[node_type](F.gelu(out))
#                 # if out.size(-1) == x_dict[node_type].size(-1):
#                     # alpha = self.skip[node_type].sigmoid()
#                 out = self.gru_layer(out, x_dict[node_type])
#                 out_dict[node_type] = out
            
#             x_dict = out_dict

#         return x_dict
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
    def __init__(self, channels, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace = True)
        )
    def forward(self, ast_nodes):
        ast_node_index, ast_node_embeddings = [], []
        for node, info in ast_nodes.items():
            node_embedding = self.embedding_layer(info['node_type_index'], info['node_sub_token_ids'], reduce_dim = 0, concat_dim = -1)
            ast_node_index.append(node)
            ast_node_embeddings.append(node_embedding)
        ast_node_embeddings = torch.stack(ast_node_embeddings, dim = 0)
        ast_node_embeddings = self.mlp(ast_node_embeddings)
        return ast_node_index, ast_node_embeddings

class ASTNodeEmbeddingFFD(nn.Module):
    def __init__(self, channels, embedding_layer, dropout = 0.1):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout)
        )
    def forward(self, ast_nodes):
        ast_node_index, ast_node_embeddings = [], []
        for node, info in ast_nodes.items():
            node_embedding = self.embedding_layer(info['node_type_index'], info['node_sub_token_ids'], reduce_dim = 0, concat_dim = -1)
            ast_node_index.append(node)
            ast_node_embeddings.append(node_embedding)
        ast_node_embeddings = torch.stack(ast_node_embeddings, dim = 0)
        ast_node_embeddings = self.mlp(ast_node_embeddings)
        return ast_node_index, ast_node_embeddings
            

class TBCNNFFDBlock(nn.Module):
    def __init__(self, embedding_layer: nn.Module, config: dict, aggr: str = 'max-pooling', pos_encoding: bool = True):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.aggr = aggr
        
        assert self.in_channels % 2 == 0, "the number of in channels must be even"
        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.pos_embed_layer = PositionalEmbedding(self.in_channels, 50)
        self.attn_w = nn.Parameter(torch.randn(self.out_channels, 1))
        self.w_t = nn.Parameter(torch.randn(self.in_channels, self.out_channels)) 
        self.w_l = nn.Parameter(torch.randn(self.in_channels, self.out_channels))
        self.w_r = nn.Parameter(torch.randn(self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels, ))
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(self.dropout_rate)
    
        self.norm_layer = nn.LayerNorm(self.out_channels)

        if self.aggr == 'attention':
            self.depth_embedding = nn.Embedding(8, 1, padding_idx = 0)        
            self.query_trans = nn.Linear(self.out_channels, self.out_channels)
            self.key_trans = nn.Linear(self.out_channels, self.out_channels)
            self.value_trans = nn.Linear(self.out_channels, self.out_channels)
            # self.control_gate = nn.Linear(2 * self.out_channels, 1)
            self.gate = nn.Parameter(torch.zeros(1))

        self.init_params()
    def init_params(self):
        nn.init.xavier_normal_(self.w_t)
        nn.init.xavier_normal_(self.w_l)
        nn.init.xavier_normal_(self.w_r)
        nn.init.xavier_normal_(self.attn_w)
    def forward(self, node_index, node_type_index, node_height, node_token_ids, children_index):
        parent_node_type_embedding = self.get_parent_type_embedding(node_type_index)
        parent_node_token_embedding = self.get_parent_token_embedding(node_token_ids)

        parent_node_embedding = torch.cat((parent_node_type_embedding, parent_node_token_embedding), dim = 2)
        if self.pos_encoding:
            parent_node_embedding = self.pos_embed_layer(parent_node_embedding, node_index)
        # children_type_embedding = self.get_children_embedding_from_parent(parent_node_type_embedding, children_index)
        # children_token_embedding = self.get_children_tokens_embedding(children_node_token_ids)
        children_embedding = self.get_children_embedding_from_parent(parent_node_embedding, children_index)
        # children_embedding = torch.cat((children_type_embedding, children_token_embedding), dim = 3)

        # parent_node_embedding = self.parent_mlp(parent_node_embedding)
        # children_embedding = self.children_mlp(children_embedding)
        node_embedding = self.conv_op(parent_node_embedding, children_embedding, children_index)
        return self.aggregate(node_embedding, node_type_index, node_height)
    def aggregate(self, node_embedding, node_type_index, node_height):
        # return torch.max(node_embedding, 1)[0]
        if self.aggr == 'attention':
            # scores = torch.matmul(node_embedding, self.attn_w)
            # node_type_index = node_type_index.unsqueeze(-1)
            # scores = torch.where(node_type_index == -1, torch.tensor(float('-inf'), device = node_embedding.device), scores)
            # attn_coefficients = F.softmax(scores, dim = 1)
            
            # tree_embeddings =  torch.sum(node_embedding * attn_coefficients, dim = 1)
            children_embedding = node_embedding[:, 1:]
            root_embedding = node_embedding[:, 0]

            children_height = node_height[:, 1:]
            
            key_embedding = self.key_trans(children_embedding)
            query_embedding = self.query_trans(root_embedding)
            value_embedding = self.value_trans(children_embedding)
            logits = torch.einsum('ij, ikj -> ik', query_embedding, key_embedding) + self.depth_embedding(children_height.long()).squeeze(-1)
            scores = F.softmax(logits, dim = 1)
            agg_children_embedding = torch.einsum('ik, ikj -> ij', scores, value_embedding)

            # control_gate = torch.sigmoid(self.control_gate(torch.cat((root_embedding, agg_children_embedding), dim = 1)))
            control_gate = torch.sigmoid(self.gate)
            tree_embeddings = control_gate * root_embedding + (1 - control_gate) * agg_children_embedding
        elif self.aggr == 'max-pooling':
            tree_embeddings = torch.max(node_embedding, dim = 1)[0]
        return tree_embeddings

    def conv_op(self, parent_node_embedding, children_embedding, children_index):
        parent_node_embedding = self.conv_step(parent_node_embedding, children_embedding, children_index, w_t = self.w_t, w_l = self.w_l, w_r = self.w_r, bias = self.bias)
        parent_node_embedding = self.norm_layer(parent_node_embedding)
        parent_node_embedding = self.relu(parent_node_embedding)
        parent_node_embedding = self.dropout(parent_node_embedding)
        # children_embedding = self.get_children_embedding_from_parent(parent_node_embedding, children_index)
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
        return self.embedding_layer.get_type_embedding(parent_node_type_index)
    def get_parent_token_embedding(self, parent_node_token_ids):
        return self.embedding_layer.get_token_embedding(parent_node_token_ids).sum(dim = 2)
    def get_children_tokens_embedding(self, children_node_token_ids):
        return self.embedding_layer.get_token_embedding(children_node_token_ids).sum(dim = 3)
    

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



class Token_TBCNNFFDBlock(nn.Module):
    def __init__(self, embedding_layer: nn.Module, config: dict, aggr: str = 'attention', pos_encoding: bool = True):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.aggr = aggr
        
        assert self.in_channels % 2 == 0, "the number of in channels must be even"
        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.pos_embed_layer = PositionalEmbedding(self.in_channels, 50)
        self.attn_w = nn.Parameter(torch.randn(self.out_channels, 1))
        self.w_t = nn.Parameter(torch.randn(self.in_channels, self.out_channels)) 
        self.w_l = nn.Parameter(torch.randn(self.in_channels, self.out_channels))
        self.w_r = nn.Parameter(torch.randn(self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels, ))
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(self.dropout_rate)
    
        self.norm_layer = nn.LayerNorm(self.out_channels)

        if self.aggr == 'attention':
            self.depth_embedding = nn.Embedding(8, 1, padding_idx = 0)        
            self.query_trans = nn.Linear(self.out_channels, self.out_channels)
            self.key_trans = nn.Linear(self.out_channels, self.out_channels)
            self.value_trans = nn.Linear(self.out_channels, self.out_channels)
            # self.control_gate = nn.Linear(2 * self.out_channels, 1)
            self.gate = nn.Parameter(torch.zeros(1))

        self.init_params()
    def init_params(self):
        nn.init.xavier_normal_(self.w_t)
        nn.init.xavier_normal_(self.w_l)
        nn.init.xavier_normal_(self.w_r)
        nn.init.xavier_normal_(self.attn_w)
    def forward(self, node_index, node_type_index, node_height, node_token_ids, token_node_ids, children_index):
        parent_node_type_embedding = self.get_parent_type_embedding(node_type_index)
        parent_node_token_embedding = self.get_parent_token_embedding(node_token_ids)

        parent_node_embedding = torch.cat((parent_node_type_embedding, parent_node_token_embedding), dim = 2)
        if self.pos_encoding:
            parent_node_embedding = self.pos_embed_layer(parent_node_embedding, node_index)
        # children_type_embedding = self.get_children_embedding_from_parent(parent_node_type_embedding, children_index)
        # children_token_embedding = self.get_children_tokens_embedding(children_node_token_ids)
        children_embedding = self.get_children_embedding_from_parent(parent_node_embedding, children_index)
        # children_embedding = torch.cat((children_type_embedding, children_token_embedding), dim = 3)

        # parent_node_embedding = self.parent_mlp(parent_node_embedding)
        # children_embedding = self.children_mlp(children_embedding)
        node_embedding = self.conv_op(parent_node_embedding, children_embedding, children_index)
        

        token_node_embeds = self.get_token_node_embeddings(node_embedding, token_node_ids)
        return self.aggregate(node_embedding, node_type_index, node_height), token_node_embeds
    def aggregate(self, node_embedding, node_type_index, node_height):
        # return torch.max(node_embedding, 1)[0]
        if self.aggr == 'attention':
            # scores = torch.matmul(node_embedding, self.attn_w)
            # node_type_index = node_type_index.unsqueeze(-1)
            # scores = torch.where(node_type_index == -1, torch.tensor(float('-inf'), device = node_embedding.device), scores)
            # attn_coefficients = F.softmax(scores, dim = 1)
            
            # tree_embeddings =  torch.sum(node_embedding * attn_coefficients, dim = 1)
            children_embedding = node_embedding[:, 1:]
            root_embedding = node_embedding[:, 0]

            children_height = node_height[:, 1:]
            
            key_embedding = self.key_trans(children_embedding)
            query_embedding = self.query_trans(root_embedding)
            value_embedding = self.value_trans(children_embedding)
            logits = torch.einsum('ij, ikj -> ik', query_embedding, key_embedding) + self.depth_embedding(children_height.long()).squeeze(-1)
            scores = F.softmax(logits, dim = 1)
            agg_children_embedding = torch.einsum('ik, ikj -> ij', scores, value_embedding)

            # control_gate = torch.sigmoid(self.control_gate(torch.cat((root_embedding, agg_children_embedding), dim = 1)))
            control_gate = torch.sigmoid(self.gate)
            tree_embeddings = control_gate * root_embedding + (1 - control_gate) * agg_children_embedding
        elif self.aggr == 'max-pooling':
            tree_embeddings = torch.max(node_embedding, dim = 1)[0]
        return tree_embeddings

    def conv_op(self, parent_node_embedding, children_embedding, children_index):
        parent_node_embedding = self.conv_step(parent_node_embedding, children_embedding, children_index, w_t = self.w_t, w_l = self.w_l, w_r = self.w_r, bias = self.bias)
        parent_node_embedding = self.norm_layer(parent_node_embedding)
        parent_node_embedding = self.relu(parent_node_embedding)
        parent_node_embedding = self.dropout(parent_node_embedding)
        # children_embedding = self.get_children_embedding_from_parent(parent_node_embedding, children_index)
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
    
    def get_token_node_embeddings(self, node_embedding, token_node_ids):
        pad_node_embeds = F.pad(node_embedding, (0, 0, 1, 0))
        batch_size, num_tokens = token_node_ids.shape
        token_node_ids = token_node_ids.unsqueeze(-1)
        batch_index = torch.arange(batch_size).view(batch_size, 1, 1).to(node_embedding.device)
        batch_index = torch.tile(batch_index, (1, num_tokens, 1))
        token_node_ids = torch.cat((batch_index, token_node_ids), dim = -1)
        return gather_nd(pad_node_embeds, token_node_ids)


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
        return self.embedding_layer.get_type_embedding(parent_node_type_index)
    def get_parent_token_embedding(self, parent_node_token_ids):
        return self.embedding_layer.get_token_embedding(parent_node_token_ids).sum(dim = 2)
    def get_children_tokens_embedding(self, children_node_token_ids):
        return self.embedding_layer.get_token_embedding(children_node_token_ids).sum(dim = 3)
    

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


class TBCNNLayer(nn.Module):
    def __init__(self, embedding_layer: nn.Module, config: dict, aggr: str = 'attention', pos_encoding: bool = True):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.aggr = aggr
        self.use_norm = config['normalization'] is not None
        
        assert self.in_channels % 2 == 0, "the number of in channels must be even"
        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.pos_embed_layer = PositionalEmbedding(self.in_channels, 50)
        self.attn_w = nn.Parameter(torch.randn(self.out_channels, 1))
        self.w_t_lst = nn.ParameterList([nn.Parameter(torch.randn(self.in_channels, self.out_channels)) for _ in range(self.num_layers)])
        self.w_l_lst = nn.ParameterList([nn.Parameter(torch.randn(self.in_channels, self.out_channels)) for _ in range(self.num_layers)])
        self.w_r_lst = nn.ParameterList([nn.Parameter(torch.randn(self.in_channels, self.out_channels)) for _ in range(self.num_layers)])
        self.bias_lst = nn.ParameterList([nn.Parameter(torch.zeros(self.out_channels, )) for _ in range(self.num_layers)])
        self.leaky_relu_lst = nn.ModuleList([nn.LeakyReLU(inplace = True) for _ in range(self.num_layers)])
        self.dropout_lst = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(self.num_layers)])
    
        self.norm_layers = nn.ModuleList([nn.LayerNorm(self.out_channels) for _ in range(self.num_layers)])
        self.input_grads = []
        if self.aggr == 'attention':
            self.depth_embedding = nn.Embedding(8, 1, padding_idx = 0)        
            self.query_trans = nn.Linear(self.out_channels, self.out_channels)
            self.key_trans = nn.Linear(self.out_channels, self.out_channels)
            self.value_trans = nn.Linear(self.out_channels, self.out_channels)
            # self.control_gate = nn.Linear(2 * self.out_channels, 1)
            self.gate = nn.Parameter(torch.zeros(1))

        self.init_params()

    def get_input_grads(self, grad):
        self.input_grads.append(grad)
    def init_params(self):
        for w_t, w_l, w_r in zip(self.w_t_lst, self.w_l_lst, self.w_r_lst):
            nn.init.xavier_normal_(w_t)
            nn.init.xavier_normal_(w_l)
            nn.init.xavier_normal_(w_r)
        nn.init.xavier_normal_(self.attn_w)
    def forward(self, node_index, node_type_index, node_height, node_token_ids, children_index):
        parent_node_type_embedding = self.get_parent_type_embedding(node_type_index)
        parent_node_token_embedding = self.get_parent_token_embedding(node_token_ids)

        parent_node_embedding = torch.cat((parent_node_type_embedding, parent_node_token_embedding), dim = 2)
        # parent_node_embedding.register_hook(self.get_input_grads)
        if self.pos_encoding:
            parent_node_embedding = self.pos_embed_layer(parent_node_embedding, node_index)
        # children_type_embedding = self.get_children_embedding_from_parent(parent_node_type_embedding, children_index)
        # children_token_embedding = self.get_children_tokens_embedding(children_node_token_ids)
        children_embedding = self.get_children_embedding_from_parent(parent_node_embedding, children_index)
        # children_embedding = torch.cat((children_type_embedding, children_token_embedding), dim = 3)

        # parent_node_embedding = self.parent_mlp(parent_node_embedding)
        # children_embedding = self.children_mlp(children_embedding)
        node_embedding = self.conv_op(parent_node_embedding, children_embedding, children_index)

        return self.aggregate(node_embedding, node_type_index, node_height)
    def aggregate(self, node_embedding, node_type_index, node_height):
        # return torch.max(node_embedding, 1)[0]
        if self.aggr == 'attention':
            # scores = torch.matmul(node_embedding, self.attn_w)
            # node_type_index = node_type_index.unsqueeze(-1)
            # scores = torch.where(node_type_index == -1, torch.tensor(float('-inf'), device = node_embedding.device), scores)
            # attn_coefficients = F.softmax(scores, dim = 1)
            
            # tree_embeddings =  torch.sum(node_embedding * attn_coefficients, dim = 1)
            children_embedding = node_embedding[:, 1:]
            root_embedding = node_embedding[:, 0]

            children_height = node_height[:, 1:]
            
            key_embedding = self.key_trans(children_embedding)
            query_embedding = self.query_trans(root_embedding)
            value_embedding = self.value_trans(children_embedding)
            logits = torch.einsum('ij, ikj -> ik', query_embedding, key_embedding) + self.depth_embedding(children_height.long()).squeeze(-1)
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
        return self.embedding_layer.get_type_embedding(parent_node_type_index)
    def get_parent_token_embedding(self, parent_node_token_ids):
        return self.embedding_layer.get_token_embedding(parent_node_token_ids).sum(dim = 2)
    def get_children_tokens_embedding(self, children_node_token_ids):
        return self.embedding_layer.get_token_embedding(children_node_token_ids).sum(dim = 3)
    

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

class HeteroGatedGCNLayer(nn.Module): 
    def __init__(self, in_channels, number_steps, etypes, use_norm = False):
        super().__init__()
        self.in_channels = in_channels
        self.number_steps = number_steps
        self.weights= nn.ModuleDict({
            etype: nn.Linear(in_channels, in_channels) for etype in etypes
        })
        self.gru_cell = nn.GRUCell(in_channels, in_channels)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.2)
    def forward(self, g, node_features):
        for step in range(self.number_steps):
            funcs = {}
            for etype in g.etypes:
                We = self.weights[etype]
                g.nodes['node'].data[f'We_{etype}'] = We(node_features)
                funcs[etype] = (fn.copy_u(f'We_{etype}', 'm'), fn.mean('m', 'h'))
            g.multi_update_all(funcs, 'sum')
            node_features =  self.gru_cell(g.nodes['node'].data['h'], node_features)
            if self.use_norm:
                node_features = self.norm(node_features)
            node_features = self.dropout(node_features)
        return node_features

class DglHGTConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 *,
                 dropout: int = 0.2,
                 use_norm : bool = False):
        """
        Reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py
        """
        super().__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            k_dict, q_dict, v_dict = {}, {}, {}
            # Iterate over node-types:
            for node_type, node_id in node_dict.items():
                k_dict[node_type] = self.k_linears[node_id](h[node_type])
                q_dict[node_type] = self.q_linears[node_id](h[node_type])
                v_dict[node_type] = self.v_linears[node_id](h[node_type])
            funcs = {}
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k = k_dict[srctype].view(-1, self.n_heads, self.d_k)
                v = v_dict[srctype].view(-1, self.n_heads, self.d_k)
                q = q_dict[dsttype].view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[srctype, etype, dsttype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata[f'v_{e_id}'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata[f't_{e_id}'] = attn_score.unsqueeze(-1)
                funcs[srctype, etype, dsttype] = fn.u_mul_e(f'v_{e_id}', f't_{e_id}', 'm'), fn.sum('m', 't')
            G.multi_update_all(funcs, cross_reducer = 'sum')
            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = F.gelu(G.nodes[ntype].data['t'].view(-1, self.out_dim))
                trans_out = self.a_linears[n_id](t)
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

class DglHGTGRUConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 time_steps,
                 *,
                 dropout: int = 0.2,
                 use_norm: bool = False):
        """
        Reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py
        """
        super().__init__()
        self.time_steps = time_steps
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        self.gru_layer = nn.GRUCell(out_dim, out_dim)
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        # self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        for _ in range(self.time_steps):
            with G.local_scope():
                node_dict, edge_dict = self.node_dict, self.edge_dict
                k_dict, q_dict, v_dict = {}, {}, {}
                # Iterate over node-types:
                for node_type, node_id in node_dict.items():
                    k_dict[node_type] = self.k_linears[node_id](h[node_type])
                    q_dict[node_type] = self.q_linears[node_id](h[node_type])
                    v_dict[node_type] = self.v_linears[node_id](h[node_type])
                funcs = {}
                for srctype, etype, dsttype in G.canonical_etypes:
                    sub_graph = G[srctype, etype, dsttype]

                    k = k_dict[srctype].view(-1, self.n_heads, self.d_k)
                    v = v_dict[srctype].view(-1, self.n_heads, self.d_k)
                    q = q_dict[dsttype].view(-1, self.n_heads, self.d_k)

                    e_id = self.edge_dict[srctype, etype, dsttype]

                    relation_att = self.relation_att[e_id]
                    relation_pri = self.relation_pri[e_id]
                    relation_msg = self.relation_msg[e_id]

                    k = torch.einsum("bij,ijk->bik", k, relation_att)
                    v = torch.einsum("bij,ijk->bik", v, relation_msg)

                    sub_graph.srcdata['k'] = k
                    sub_graph.dstdata['q'] = q
                    sub_graph.srcdata[f'v_{e_id}'] = v

                    sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                    attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                    attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                    sub_graph.edata[f't_{e_id}'] = attn_score.unsqueeze(-1)
                    funcs[srctype, etype, dsttype] = fn.u_mul_e(f'v_{e_id}', f't_{e_id}', 'm'), fn.sum('m', 't')
                G.multi_update_all(funcs, cross_reducer = 'sum')
                new_h = {}
                for ntype in G.ntypes:
                    '''
                        Step 3: Target-specific Aggregation
                        x = norm( W[node_type] * gelu( Agg(x) ) + x )
                    '''
                    n_id = node_dict[ntype]
                    # alpha = torch.sigmoid(self.skip[n_id])
                    t = F.gelu(G.nodes[ntype].data['t'].view(-1, self.out_dim))
                    trans_out = self.a_linears[n_id](t)
                    # trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                    trans_out = self.gru_layer(trans_out, h[ntype])
                    if self.use_norm:
                        new_h[ntype] = self.norms[n_id](trans_out)
                    else:
                        new_h[ntype] = trans_out
                h = new_h
        return h

class DglHGTFFDConvBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 d_ff,
                 node_dict,
                 edge_dict,
                 n_heads,
                 *,
                 dropout: int = 0.2):
        """
        Reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py
        """
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        self.ffd_layers = nn.ModuleList()
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            self.ffd_layers.append(PositionwiseFeedForward(out_dim, d_ff))
            self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        # self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            k_dict, q_dict, v_dict = {}, {}, {}
            # Iterate over node-types:
            for node_type, node_id in node_dict.items():
                k_dict[node_type] = self.k_linears[node_id](h[node_type])
                q_dict[node_type] = self.q_linears[node_id](h[node_type])
                v_dict[node_type] = self.v_linears[node_id](h[node_type])
            funcs = {}
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k = k_dict[srctype].view(-1, self.n_heads, self.d_k)
                v = v_dict[srctype].view(-1, self.n_heads, self.d_k)
                q = q_dict[dsttype].view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[srctype, etype, dsttype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata[f'v_{e_id}'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata[f't_{e_id}'] = attn_score.unsqueeze(-1)
                funcs[srctype, etype, dsttype] = fn.u_mul_e(f'v_{e_id}', f't_{e_id}', 'm'), fn.sum('m', 't')
            G.multi_update_all(funcs, cross_reducer = 'sum')
            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                # alpha = torch.sigmoid(self.skip[n_id])
                t = torch.relu(G.nodes[ntype].data['t'].view(-1, self.out_dim))
                trans_out = self.a_linears[n_id](t)
                x = trans_out + h[node_type]
                x = self.drop(x)
                x = self.norms[n_id](x)
                x = self.ffd_layers[n_id](x)
                new_h[node_type] = x
        return new_h

class HGTLayer(nn.Module):
    def __init__(self, config, metadata, dgl_format, func):
        super().__init__()
        self.in_channels = config['in_channels']
        self.num_graph_heads = config['num_graph_heads']
        self.num_graph_steps = config['num_graph_steps']

        self.linear_trans = nn.ModuleDict()
        self.dgl_format = dgl_format
        self.use_norm = config.get('normalization', None)

        if 'feedforward_decoder_channels' in config:
            self.ffd_channels = config['feedforward_decoder_channels']
        # for node_type in metadata[0]:
        #     self.linear_trans[node_type] = nn.Linear(self.in_channels, self.in_channels)

        if dgl_format:
            if func == 'linear':
                self.convs = nn.ModuleList([
                    DglHGTConv(self.in_channels, self.in_channels, metadata[0], metadata[1], self.num_graph_heads, use_norm = self.use_norm)
                    for _ in range(self.num_graph_steps)
                ])
            elif func == 'gru':
                self.convs = nn.ModuleList([
                    DglHGTGRUConv(self.in_channels, self.in_channels, metadata[0], metadata[1], self.num_graph_heads, timestep, use_norm = self.use_norm)
                    for timestep in config['time_steps']
                ])
            elif func == 'ffd':
                self.convs = nn.ModuleList([
                    DglHGTFFDConvBlock(self.in_channels, self.in_channels, self.ffd_channels, metadata[0], metadata[1], self.num_graph_heads)
                    for _ in range(self.num_graph_steps)
                ])
        else:
            if func == 'linear':
                self.convs = nn.ModuleList([
                    HGTConv(self.in_channels, self.in_channels, metadata, self.num_graph_heads, group = 'sum')
                    for _ in range(self.num_graph_steps)
                ])
            elif func == 'gru':
                self.convs = nn.ModuleList([
                    HGTGRUConv(self.in_channels, self.in_channels, metadata, timestep, self.num_graph_heads, group = 'sum')
                   for timestep in config['time_steps']
                ])
    def forward(self, x_dict, graphs = None, edge_index_dict = None):
        # for node_type, x in x_dict.items():
        #     x_dict[node_type] = F.relu_(self.linear_trans[node_type](x))
        if self.dgl_format:
            for conv in self.convs:
                x_dict = conv(graphs, x_dict)
        else:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
        return x_dict


class AttentionGraphLevel(nn.Module):
    def __init__(self, channels, aggr):
        super().__init__()
        self.aggr = aggr
        if self.aggr == 'attention':
            self.attn_w = nn.Parameter(torch.randn(channels, 1))
            # self.w_g = nn.Parameter(torch.randn(channels, channels))
            # self.w_cg = nn.Parameter(torch.randn(channels, channels))
            # self.w_gg = nn.Parameter(torch.randn(channels, channels))
            nn.init.xavier_normal_(self.attn_w)
            # nn.init.xavier_normal_(self.w_g)
            # nn.init.xavier_normal_(self.w_cg)
            # nn.init.xavier_normal_(self.w_gg)
            self.w = nn.Linear(channels, 1)
            self.f = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(inplace = True),
                nn.Linear(channels, 1)
            )
        elif self.aggr == 'conv-attention':
            self.conv = nn.Conv1d(2, 1, kernel_size = 7, padding = 3)
    def forward(self, node_embeddings, num_nodes, latent_embeddings):
        groups = self.get_groups(num_nodes).to(node_embeddings.device)

        if self.aggr == 'attention':
            context_embeddings = node_embeddings[groups]
            latent_embeddings = latent_embeddings[groups]
            alpha = torch.sigmoid(self.w(context_embeddings))
            attn_w = alpha * self.f(latent_embeddings) + (1 - alpha) * self.attn_w.t()
            # graph_embeddings = scatter_mean(node_embeddings, groups, dim = 0)
            # s = torch.sigmoid(torch.matmul(torch.matmul(context_embeddings, self.w_g), self.attn_w))
            # attn_w = (1 - s) * (context_embeddings @ self.w_cg) + s * (self.attn_w.t() @ self.w_gg)
            scores = torch.sum(node_embeddings * attn_w, dim = 1, keepdim = True)
            scores = scores - torch.max(scores, 0, keepdim = True)[0]
            scores = torch.exp(scores)
            denominator = scatter_add(scores, groups, dim = 0)[groups]
            scores = scores / denominator
            graph_embeddings = scatter_add(node_embeddings * scores, groups, dim = 0)
        elif self.aggr == 'conv-attention':
            embeddings = []
            for index, num_node in enumerate(num_nodes):
                embeddings.append(node_embeddings[[index] * num_node])
            embs = nn.utils.rnn.pad_sequence(embeddings, batch_first = True).transpose(1, 2)
            out_avg = torch.sum(embs, 1, True) / torch.Tensor(num_nodes, device = node_embeddings.device).unsqueeze(-1).unsqueeze(-1)
            out_max = torch.max(embs, 1, True)[0]
            out = torch.cat((out_avg, out_max), dim = 1)
            out = self.conv(out)
            mask = torch.zeros(embs.shape[0], 1, embs.shape[2], device = node_embeddings.device)
            for i, num_node in enumerate(num_nodes):
                mask[i, 0, num_node:] = float('-inf')
            out = out + mask
            scores = F.softmax(out, dim = 2)
            # out = torch.sigmoid(out)
            return torch.sum(embs * scores, dim = 2)
        elif self.aggr == 'max-pooling':
            graph_embeddings = scatter_max(node_embeddings, groups, dim = 0)[0]
        elif self.aggr == 'avg-pooling':
            graph_embeddings = scatter_mean(node_embeddings, groups, dim = 0)
        return graph_embeddings
    def get_groups(self, num_nodes):
        groups = []
        for index, num_node in enumerate(num_nodes):
            groups.extend([index] * num_node)
        return torch.tensor(groups)

if __name__ == "__main__":
    TBCNNLayer(10, 10, 3)
        