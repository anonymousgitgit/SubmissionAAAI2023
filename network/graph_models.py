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


class HierarchicalGatedModel(nn.Module):
    def __init__(self, config: dict, *, tree_aggr: str, graph_aggr: str, pos_encoding: bool):
        super().__init__()
        self.node_embedding_layer = NodeEmbedding(config)
        self.ast_embedding_layer = ASTNodeEmbedding(config['out_channels'], self.node_embedding_layer)
        self.tbcnn_layer = TBCNNLayer(self.node_embedding_layer, config, tree_aggr, pos_encoding)
        self.gated_gcn_lst = nn.ModuleList([HeteroGatedGCNLayer(config['out_channels'], config['time_steps'][i], config['etypes'], config['normalization'] is not None) for i, _ in enumerate(range(len(config['time_steps'])))])
        self.fc_layer = nn.Linear(config['out_channels'], config['n_classes'])
        self.attn_graph_layer = AttentionGraphLevel(config['out_channels'], graph_aggr)
        
        self.w = nn.Linear(config['out_channels'], config['out_channels'])
        self.tree_aggr = tree_aggr
        self.graph_aggr = graph_aggr
        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.central_embedding = nn.Embedding(150, config['out_channels'])
    def forward(self, num_nodes, ast_node_index, tree, graphs, in_degrees):
        ast_node_index, ast_node_embeddings = self.ast_embedding_layer(ast_node_index)
        if self.training:
            node_embeddings, batch_tree_index = [], []
            sizes = list(tree.keys())
            random.shuffle(sizes)
            for size in sizes:
                batch = tree[size]
                each_bucket_embeddings = self.tbcnn_layer(batch['batch_node_index'], batch['batch_node_type_id'], batch['batch_node_height'], batch['batch_node_sub_tokens_id'], batch['batch_children_index'])
                node_embeddings.append(each_bucket_embeddings)
                batch_tree_index.extend(batch['batch_tree_index'])
            node_embeddings = torch.cat(node_embeddings, axis = 0)
        else:
            batch_tree_index = tree['batch_tree_index']
            node_embeddings = self.tbcnn_layer(tree['batch_node_index'], tree['batch_node_type_id'], tree['batch_node_height'], tree['batch_node_sub_tokens_id'], tree['batch_children_index'])
        
        embeddings = torch.cat((ast_node_embeddings, node_embeddings), dim = 0)
        # ensure that after merging, the order of nodes matches the order in graph
        order_indices = np.argsort(ast_node_index + batch_tree_index)
        # print('a', order_indices.shape)
        if self.pos_encoding:
            node_embeddings = embeddings[order_indices] + self.central_embedding(in_degrees['node'].long())
        else:
            node_embeddings = embeddings[order_indices]
        # node_embeddings = torch.tanh(self.w(node_embeddings))
        for gated_gcn in self.gated_gcn_lst:
            node_embeddings = gated_gcn(graphs, node_embeddings)
        graph_embeddings = self.attn_graph_layer(node_embeddings, num_nodes['num_nodes'], node_embeddings[num_nodes['last_stmts']])
        logits = self.fc_layer(graph_embeddings)
        return embeddings, logits


class HierarchicalHGTModel(nn.Module):
    def __init__(self, config, metadata, dgl_format, func, num_node_types, *, tree_aggr, graph_aggr, pos_encoding):
        super().__init__()
        self.dgl_format = dgl_format
        self.tree_aggr = tree_aggr
        self.graph_aggr = graph_aggr
        self.node_embedding_layer = NodeEmbedding(config)
        self.ast_embedding_layer = ASTNodeEmbedding(config['out_channels'], self.node_embedding_layer)
        self.tbcnn_layer = TBCNNLayer(self.node_embedding_layer, config, tree_aggr, pos_encoding)
        self.hgt_graph_layer = HGTLayer(config, metadata, dgl_format, func)
        self.fc_layer = nn.Linear(num_node_types * config['out_channels'], config['n_classes'])
        self.num_node_types = num_node_types
        self.attn_graph_layer = AttentionGraphLevel(config['out_channels'], graph_aggr)
        self.attn_graph_layer1 = AttentionGraphLevel(config['out_channels'], graph_aggr)
        self.pos_encoding = pos_encoding
        self.input_grads = None
        self.batch_node_tokens = []
        self.batch_node_types = []
        if self.pos_encoding:
            self.central_embedding = nn.Embedding(150, config['out_channels'])
        # self.w = nn.Linear(config['out_channels'], config['out_channels'])
        self.tree_indices = []
    def get_input_grads(self, grad):
        self.input_grads = grad
    def forward(self, num_nodes, ast_node_index, tree, graphs, in_degrees):
        ast_node_index, ast_node_embeddings = self.ast_embedding_layer(ast_node_index)
        node_embeddings, batch_tree_index = [], []
        sizes = list(tree.keys())
        if self.training:
            random.shuffle(sizes)
        for size in sizes:
            batch = tree[size]
            # self.batch_node_tokens.append(batch['batch_node_token'])
            # self.batch_node_types.append(batch['batch_node_type'])
            # self.tree_indices.append(batch['batch_tree_index'])
            each_bucket_embeddings = self.tbcnn_layer(batch['batch_node_index'], batch['batch_node_type_id'], batch['batch_node_height'], batch['batch_node_sub_tokens_id'], batch['batch_children_index'])
            node_embeddings.append(each_bucket_embeddings)
            batch_tree_index.extend(batch['batch_tree_index'])
        stmt_node_embeddings = torch.cat(node_embeddings, axis = 0)
        
        # embeddings = torch.cat((ast_node_embeddings, node_embeddings), dim = 0)
        if self.dgl_format:
            if self.num_node_types == 1:
                embeddings = torch.cat((ast_node_embeddings, stmt_node_embeddings), dim = 0)
                # ensure that after merging, the order of nodes matches the order in graph
                order_indices = np.argsort(ast_node_index + batch_tree_index)
                # print('a', order_indices.shape)
                if self.pos_encoding:
                    node_embeddings = embeddings[order_indices] + self.central_embedding(in_degrees['node'].long())
                else:
                    node_embeddings = embeddings[order_indices]
                # self.input = node_embeddings
                # print(self.input)
                # node_embeddings.register_hook(self.get_input_grads)
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
            all_node_embeddings = self.hgt_graph_layer(data, graphs = graphs)
        else:
            ast_order_node_embeddings = ast_node_embeddings[np.argsort(ast_node_index)]
            stmt_order_node_embeddings = stmt_node_embeddings[np.argsort(batch_tree_index)]
            graphs['ast_node'].x = ast_order_node_embeddings
            graphs['stmt_node'].x = stmt_order_node_embeddings
            # print(graphs.x_dict)
            all_node_embeddings = self.hgt_graph_layer(graphs.x_dict, edge_index_dict = graphs.edge_index_dict)
        if self.num_node_types == 2:
            stmt_node_embeddings = all_node_embeddings['stmt_node']
        else:
            graph_embeddings = self.attn_graph_layer(all_node_embeddings['node'], num_nodes['num_nodes'], all_node_embeddings['node'][num_nodes['last_stmts']]) 
        logits = self.fc_layer(graph_embeddings)
        return all_node_embeddings, logits


class HierarchicalHGT2Seq(nn.Module):
    def __init__(self, config, metadata, dgl_format, func, num_node_types, *, tree_aggr, graph_aggr, pos_encoding, sos_index):
        super().__init__()
        self.dgl_format = dgl_format
        self.tree_aggr = tree_aggr
        self.graph_aggr = graph_aggr
        self.max_seq_length = config['max_seq_length']
        self.sos_index = sos_index
        self.node_embedding_layer = NodeEmbedding(config)
        self.ast_embedding_layer = ASTNodeEmbedding(config['out_channels'], self.node_embedding_layer)
        self.tbcnn_layer = TBCNNLayer(self.node_embedding_layer, config, tree_aggr, pos_encoding)
        self.hgt_graph_layer = HGTLayer(config, metadata, dgl_format, func)
        self.fc_layer = nn.Linear(num_node_types * config['out_channels'], config['n_classes'])
        self.num_node_types = num_node_types
        self.attn_graph_layer = AttentionGraphLevel(config['out_channels'], graph_aggr)
        self.attn_graph_layer1 = AttentionGraphLevel(config['out_channels'], graph_aggr)
        # self.decoder = DecoderRNN(config['out_channels'], config['vocab_comment_size'])
        self.decoder = AttnDecoderRNN('general', config['out_channels'], config['vocab_comment_size'])
        self.pos_encoding = pos_encoding
        # self.context_mlp = nn.Sequential(
        #     nn.Linear(config['out_channels'], self.decoder.hidden_size),
        #     nn.ReLU(inplace = True)
        # )
        self.norm = nn.LayerNorm(config['out_channels'])
        if self.pos_encoding:
            self.central_embedding = nn.Embedding(150, config['out_channels'])
    def forward(self, num_nodes, ast_node_index, tree, graphs, in_degrees, stmt_ids, label_word_ids = None):
        ast_node_index, ast_node_embeddings = self.ast_embedding_layer(ast_node_index)
        sizes = list(tree.keys())
        node_embeddings, batch_tree_index = [], []
        if self.training:
            random.shuffle(sizes)
        for size in sizes:
            batch = tree[size]
            each_bucket_embeddings = self.tbcnn_layer(batch['batch_node_index'], batch['batch_node_type_id'], batch['batch_node_height'], batch['batch_node_sub_tokens_id'], batch['batch_children_index'])
            node_embeddings.append(each_bucket_embeddings)
            batch_tree_index.extend(batch['batch_tree_index'])
        stmt_node_embeddings = torch.cat(node_embeddings, axis = 0)
        
        
        # embeddings = torch.cat((ast_node_embeddings, node_embeddings), dim = 0)
        if self.dgl_format:
            if self.num_node_types == 1:
                embeddings = torch.cat((ast_node_embeddings, stmt_node_embeddings), dim = 0)
                # ensure that after merging, the order of nodes matches the order in graph
                order_indices = np.argsort(ast_node_index + batch_tree_index)
                # print('a', order_indices.shape)
                if self.pos_encoding:
                    node_embeddings = embeddings[order_indices] + self.central_embedding(in_degrees['node'].long())
                else:
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
            all_node_embeddings = self.hgt_graph_layer(data, graphs = graphs)
        else:
            ast_order_node_embeddings = ast_node_embeddings[np.argsort(ast_node_index)]
            stmt_order_node_embeddings = stmt_node_embeddings[np.argsort(batch_tree_index)]
            graphs['ast_node'].x = ast_order_node_embeddings
            graphs['stmt_node'].x = stmt_order_node_embeddings
            # print(graphs.x_dict)
            all_node_embeddings = self.hgt_graph_layer(graphs.x_dict, edge_index_dict = graphs.edge_index_dict)
        
        if self.num_node_types == 2:
            stmt_node_embeddings = all_node_embeddings['stmt_node']
            ast_node_embeddings = all_node_embeddings['ast_node']
            latent_embeddings = stmt_node_embeddings[num_nodes['last_stmts']]
            graph_stmt_embeddings = self.attn_graph_layer(stmt_node_embeddings, num_nodes['stmt_nodes'], latent_embeddings)
            graph_ast_embeddings = self.attn_graph_layer1(ast_node_embeddings, num_nodes['ast_nodes'], latent_embeddings)
            graph_embeddings = torch.cat((graph_stmt_embeddings, graph_ast_embeddings), dim = -1)
        else:
            graph_embeddings = self.attn_graph_layer(all_node_embeddings['node'], num_nodes['num_nodes'], all_node_embeddings['node'][num_nodes['last_stmts']]) 
        
        
        encoder_outputs = []
        src_padding_mask = []
        if self.num_node_types == 2:
            graphs.nodes['stmt_node'].data['ft'] = all_node_embeddings['stmt_node']

            for graph in dgl.unbatch(graphs):
                embs = graph.nodes['stmt_node'].data['ft']
                encoder_outputs.append(embs)
                src_padding_mask.append(torch.tensor([0] * embs.shape[0]).unsqueeze(-1))
        else:
            graphs.nodes['node'].data['ft'] = all_node_embeddings['node']
            for graph, stmt in zip(dgl.unbatch(graphs), stmt_ids):
                embs = graph.nodes['node'].data['ft'][stmt]
                encoder_outputs.append(embs)
                src_padding_mask.append(torch.tensor([0] * embs.shape[0]).unsqueeze(-1))

        src_padding_mask = nn.utils.rnn.pad_sequence(src_padding_mask, batch_first = True, padding_value = 1).squeeze(-1).to(encoder_outputs[0].device).float()
        
        src_padding_mask = src_padding_mask.masked_fill(src_padding_mask == 1, float('-inf'))
        pad_encoder_outputs = nn.utils.rnn.pad_sequence(encoder_outputs, batch_first = True, padding_value = 0)
        
        decode_input = torch.tensor([self.sos_index], device = ast_node_embeddings.device).repeat(graph_embeddings.shape[0])
        # if graph_embeddings.shape[1] != self.decoder.hidden_size:
        #     decode_hidden = self.context_mlp(graph_embeddings)
        # else:
        decode_hidden = self.norm(graph_embeddings).unsqueeze(0)
        outputs = []
        if self.training:
            assert label_word_ids is not None
            label_word_ids = label_word_ids[:, 1:]
            use_teacher_forcing = True if random.random() < 0.5 else False
            for timestep in range(label_word_ids.shape[1]):
                decode_output, decode_hidden = self.decoder(decode_input, decode_hidden, pad_encoder_outputs, src_padding_mask)
                outputs.append(decode_output)
                decode_input = label_word_ids[:, timestep] if use_teacher_forcing else decode_output.argmax(2).squeeze(1)
            else:
                decode_output, decode_hidden = self.decoder(decode_input, decode_hidden, pad_encoder_outputs, src_padding_mask)
                outputs.append(decode_output)
        else:
            for timestep in range(self.max_seq_length):
                decode_output, decode_hidden = self.decoder(decode_input, decode_hidden, pad_encoder_outputs, src_padding_mask)
                outputs.append(decode_output)
                decode_input = decode_output.argmax(2).squeeze(1)
            
        outputs = torch.cat(outputs, dim = 1).transpose(1, 2)

        return outputs
class HierarchicalHGTTransformer(nn.Module):
    def __init__(self, config, metadata, dgl_format, func, num_node_types, *, tree_aggr, pos_encoding):
        super().__init__()
        self.dgl_format = dgl_format
        self.tree_aggr = tree_aggr
        self.max_seq_length = config['max_seq_length']
        self.node_embedding_layer = NodeEmbedding(config)
        self.ast_embedding_layer = ASTNodeEmbeddingFFD(config['out_channels'], self.node_embedding_layer)
        self.tbcnn_layer = TBCNNFFDBlock(self.node_embedding_layer, config, tree_aggr, pos_encoding)
        self.hgt_graph_layer = HGTLayer(config, metadata, dgl_format, func)
        self.num_node_types = num_node_types
        self.pos_encoding = pos_encoding
        
        self.tgt_embedder = Embedder(config['vocab_comment_size'], config['out_channels'], PAD_IDX)
        decoder_layer = nn.TransformerDecoderLayer(config['out_channels'], config['num_heads_decoder'], config['feedforward_decoder_channels'], dropout = config['dropout_decoder'], batch_first = True, norm_first = False)
        layer_norm_decoder = nn.LayerNorm(config['out_channels'])
        self.decoder = nn.TransformerDecoder(decoder_layer, config['num_decoder_layers'], norm = layer_norm_decoder)
        self.generator = nn.Linear(config['out_channels'], config['vocab_comment_size'])
        if self.pos_encoding:
            self.central_embedding = nn.Embedding(150, config['out_channels'])
    def forward(self, num_nodes, ast_node_index, tree, graphs, in_degrees, stmt_ids, label_word_ids = None):
        ast_node_index, ast_node_embeddings = self.ast_embedding_layer(ast_node_index)
        sizes = list(tree.keys())
        node_embeddings, batch_tree_index = [], []
        if self.training:
            random.shuffle(sizes)
        for size in sizes:
            batch = tree[size]
            each_bucket_embeddings = self.tbcnn_layer(batch['batch_node_index'], batch['batch_node_type_id'], batch['batch_node_height'], batch['batch_node_sub_tokens_id'], batch['batch_children_index'])
            node_embeddings.append(each_bucket_embeddings)
            batch_tree_index.extend(batch['batch_tree_index'])
        stmt_node_embeddings = torch.cat(node_embeddings, axis = 0)
        
        
        # embeddings = torch.cat((ast_node_embeddings, node_embeddings), dim = 0)
        if self.num_node_types == 1:
            embeddings = torch.cat((ast_node_embeddings, stmt_node_embeddings), dim = 0)
            # ensure that after merging, the order of nodes matches the order in graph
            order_indices = np.argsort(ast_node_index + batch_tree_index)
            # print('a', order_indices.shape)
            if self.pos_encoding:
                node_embeddings = embeddings[order_indices] + self.central_embedding(in_degrees['node'].long())
            else:
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
        all_node_embeddings = self.hgt_graph_layer(data, graphs = graphs)
        outputs = []
        src_padding_mask = []
        if self.num_node_types == 2:
            graphs.nodes['stmt_node'].data['ft'] = all_node_embeddings['stmt_node']

            for graph in dgl.unbatch(graphs):
                embs = graph.nodes['stmt_node'].data['ft']
                outputs.append(embs)
                src_padding_mask.append(torch.tensor([0] * embs.shape[0]).unsqueeze(-1))
        else:
            graphs.nodes['node'].data['ft'] = all_node_embeddings['node']
            for graph, stmt in zip(dgl.unbatch(graphs), stmt_ids):
                embs = graph.nodes['node'].data['ft'][stmt]
                outputs.append(embs)
                src_padding_mask.append(torch.tensor([0] * embs.shape[0]).unsqueeze(-1))
        
        src_padding_mask = nn.utils.rnn.pad_sequence(src_padding_mask, batch_first = True, padding_value = 1).squeeze(-1).to(outputs[0].device)
        
        pad_outputs = nn.utils.rnn.pad_sequence(outputs, batch_first = True, padding_value = 0)
        tgt_mask, tgt_padding_mask = create_target_mask(label_word_ids, padding_idx = PAD_IDX)
        
        # TODO: multiply by sqrt(dim) 
        tgt_initial_embeds = self.tgt_embedder(label_word_ids)
        out_decoder = self.decoder(tgt_initial_embeds, pad_outputs, 
                    tgt_mask = tgt_mask, 
                    tgt_key_padding_mask = tgt_padding_mask,
                    memory_key_padding_mask = src_padding_mask)
        out_generator = self.generator(out_decoder).transpose(1, 2)

        # outputs = torch.cat(outputs, dim = 1).transpose(1, 2)

        return out_generator

