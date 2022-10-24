from cmath import e
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import dgl
import random

from network.graph_layers import *


from constants import *
from modules.copy_generator import CopyGenerator
from modules.position_ffn import FeedForward
from modules.util_class import LayerNorm
from modules.utils import sequence_mask, collapse_copy_scores
from network.transformer import TransformerDecoder
from constants.vocab import PAD_IDX
from modules.global_attention import GlobalAttention
from modules.cross_attention import CrossAttention

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.input_size = config['out_channels']

        d_k = d_v = self.input_size // config['num_heads_decoder']
        self.transformer = TransformerDecoder(
            num_layers = config['num_decoder_layers'],
            d_model = self.input_size,
            heads = config['num_heads_decoder'],
            d_k = d_k,
            d_v = d_v,
            d_ff = config['feedforward_decoder_channels'],
            dropout = config['dropout_decoder'],
            coverage_attn = config.get('coverage_attn', None)
        )

        # if args.reload_decoder_state:
        #     state_dict = torch.load(
        #         args.reload_decoder_state, map_location=lambda storage, loc: storage
        #     )
        #     self.decoder.load_state_dict(state_dict)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def init_decoder(self,
                     src_lens,
                     max_src_len):
        return self.transformer.init_state(src_lens, max_src_len)

    def decode(self,
               tgt_words,
               tgt_emb,
               memory_bank,
               state,
               step=None,
               layer_wise_coverage = None):

        decoder_outputs, attns = self.transformer(tgt_words,
                                                    tgt_emb,
                                                    memory_bank,
                                                    state,
                                                    step=step,
                                                    layer_wise_coverage = layer_wise_coverage)

        return decoder_outputs, attns

    def forward(self,
                memory_bank,
                memory_len,
                tgt_pad_mask,
                tgt_emb):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]
        state = self.init_decoder(memory_len, max_mem_len)
        return self.decode(tgt_pad_mask, tgt_emb, memory_bank, state)

class MaskedHGTTransformer(nn.Module):
    def __init__(self, config, metadata, tgt_dict, dgl_format, func, num_node_types, *, tree_aggr, pos_encoding, apply_copy):
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
        self.apply_copy = apply_copy
        self.tgt_dict = tgt_dict

        self.decoder = Decoder(config)
        
        self.tgt_embedder = Embedder(config['vocab_comment_size'], config['out_channels'], PAD_IDX)
        self.generator = nn.Linear(config['out_channels'], config['vocab_comment_size'])
        if self.pos_encoding:
            self.central_embedding = nn.Embedding(150, config['out_channels'])
        
        if self.apply_copy:
            self.copy_attn = GlobalAttention(dim = config['out_channels'],
                                            attn_type = config['attn_type'])
            self.copy_generator = CopyGenerator(config['out_channels'], tgt_dict, self.generator)
    def forward(self, num_nodes, ast_node_index, tree, graphs, in_degrees, stmt_ids, extra_src_info, label_word_ids = None, **kwargs):
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
        src_map = extra_src_info['src_map']
        src_vocabs = extra_src_info['src_vocabs']
        alignments = extra_src_info['alignments']
        src_token_ids_vocab = extra_src_info['src_token_ids_vocab']
        src_lengths = extra_src_info['src_lengths']
        
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
        # print('data', data['node'].shape, np.unique(ast_node_index + batch_tree_index).shape)
        all_node_embeddings = self.hgt_graph_layer(data, graphs = graphs)
        outputs = []
        stmt_lengths = []
        if self.num_node_types == 2:
            graphs.nodes['stmt_node'].data['ft'] = all_node_embeddings['stmt_node']

            for graph in dgl.unbatch(graphs):
                embs = graph.nodes['stmt_node'].data['ft']
                outputs.append(embs)
                stmt_lengths.append(embs.shape[0])
        else:
            graphs.nodes['node'].data['ft'] = all_node_embeddings['node']
            for graph, stmt in zip(dgl.unbatch(graphs), stmt_ids):
                embs = graph.nodes['node'].data['ft']
                outputs.append(embs)
                stmt_lengths.append(embs.shape[0])
        
        memory_bank = nn.utils.rnn.pad_sequence(outputs, batch_first = True, padding_value = 0)
        enc_outputs = memory_bank

        
        stmt_lengths = torch.tensor(stmt_lengths).to(memory_bank.device)
        # tgt_mask, tgt_padding_mask = create_target_mask(label_word_ids, padding_idx = PAD_IDX)
        if self.training:
            tgt_lengths = torch.count_nonzero(label_word_ids != PAD_IDX, dim = 1)
            tgt_initial_embeds = self.tgt_embedder(label_word_ids)
            
            tgt_pad_mask = ~sequence_mask(tgt_lengths, max_len = tgt_initial_embeds.shape[1])
           
            layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                    stmt_lengths,
                                                    tgt_pad_mask,
                                                    tgt_initial_embeds
                                                )
            decoder_outputs = layer_wise_dec_out[-1]

            if self.apply_copy:
                src_token_embeds = self.tgt_embedder(src_token_ids_vocab)
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                src_token_embeds,
                                                memory_lengths = src_lengths,
                                                softmax_weights = False)
                attn_copy = F.softmax(copy_score, dim = -1)
                out_generator = self.copy_generator(decoder_outputs, attn_copy, src_map)
                out_generator = out_generator[:, :-1].contiguous()
            else:
                out_generator = self.generator(decoder_outputs[:, :-1]).transpose(1, 2)

            return out_generator
        else:
            kwargs['source_vocab'] = extra_src_info['src_vocabs']
            blank, fill = collapse_copy_scores(self.tgt_dict, extra_src_info['src_vocabs'])
            kwargs['fill'] = fill
            kwargs['blank'] = blank
            src_token_embeds = self.tgt_embedder(src_token_ids_vocab) if self.apply_copy else None
            self.layer_wise_attn = False
            return self.decode(memory_bank, 
                            src_lengths,
                            src_map,
                            alignments,
                            stmt_lengths,
                            src_token_embeds, 
                            **kwargs)
    def decode(self,
            memory_bank,
            code_len,
            src_map,
            alignment,
            stmt_lengths,
            src_token_embeds,
            **kwargs):

        params = dict()
        params['memory_bank'] = memory_bank
        params['layer_wise_outputs'] = None
        params['src_len'] = code_len
        params['stmt_len'] = stmt_lengths
        params['source_vocab'] = kwargs['source_vocab']
        params['src_map'] = src_map
        params['tgt_dict'] = self.tgt_dict
        params['max_len'] = self.max_seq_length
        params['fill'] = kwargs['fill']
        params['src_token_embeds'] = src_token_embeds
        params['blank'] = kwargs['blank']

        dec_preds, attentions, _ = self.__generate_sequence(params, choice='greedy')
        dec_preds = torch.stack(dec_preds, dim=1)
        # attentions: batch_size x tgt_len x num_heads x src_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_preds,
            'memory_bank': memory_bank,
            'attentions': attentions
        }

    def __generate_sequence(self,
                            params,
                            choice='greedy',
                            tgt_words=None):
        src_token_embeds = params['src_token_embeds']
        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([SOS_IDX])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1

        dec_preds = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]
        dec_states = self.decoder.init_decoder(params['stmt_len'], max_mem_len)

        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn \
            else params['memory_bank']
        attns = {"coverage": None}
        # +1 for <EOS> token
        for idx in range(params['max_len']):
            # tgt = self.embedder(tgt_words,
            #                     mode='decoder',
            #                     step=idx)
            tgt = self.tgt_embedder(tgt_words, step = idx)
            tgt_pad_mask = tgt_words.data.eq(PAD_IDX)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            if self.apply_copy:
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                  src_token_embeds,
                                                  memory_lengths=params['src_len'],
                                                  softmax_weights=False)

                attn_copy = F.softmax(copy_score, dim=-1)
                prediction = self.copy_generator(decoder_outputs,
                                                 attn_copy,
                                                 params['src_map'])
                prediction = prediction.squeeze(1)
                for b in range(prediction.size(0)):
                    if params['blank'][b]:
                        blank_b = torch.LongTensor(params['blank'][b])
                        fill_b = torch.LongTensor(params['fill'][b])
                        if use_cuda:
                            blank_b = blank_b.cuda()
                            fill_b = fill_b.cuda()
                        prediction[b].index_add_(0, fill_b,
                                                 prediction[b].index_select(0, blank_b))
                        prediction[b].index_fill_(0, blank_b, 1e-10)

            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = F.softmax(prediction, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))

            words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])
            words = [params['tgt_dict'].get(w, UNK_IDX) for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt_words = words.unsqueeze(1)

        return dec_preds, attentions, dec_log_probs

    def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict.inverse[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words
class MaskedHGTTransformer(nn.Module):
    def __init__(self, config, metadata, tgt_dict, dgl_format, func, num_node_types, *, tree_aggr, pos_encoding, apply_copy, lang = 'c'):
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
        self.apply_copy = apply_copy
        self.tgt_dict = tgt_dict
        if lang in ['c', 'cpp']:
            from network._decoder import Embedder
        else:
            from network.decoder import Embedder
        self.decoder = Decoder(config)
        self.tgt_embedder = Embedder(config['vocab_comment_size'], config['out_channels'], PAD_IDX)
        self.generator = nn.Linear(config['out_channels'], config['vocab_comment_size'])
        if self.pos_encoding:
            self.central_embedding = nn.Embedding(150, config['out_channels'])
        
        if self.apply_copy:
            self.copy_attn = GlobalAttention(dim = config['out_channels'],
                                            attn_type = config['attn_type'])
            self.copy_generator = CopyGenerator(config['out_channels'], tgt_dict, self.generator)
    def forward(self, num_nodes, ast_node_index, tree, graphs, in_degrees, stmt_ids, extra_src_info, label_word_ids = None, **kwargs):
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
        src_map = extra_src_info['src_map']
        src_vocabs = extra_src_info['src_vocabs']
        alignments = extra_src_info['alignments']
        src_token_ids_vocab = extra_src_info['src_token_ids_vocab']
        src_lengths = extra_src_info['src_lengths']
        
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
        # print('data', data['node'].shape, np.unique(ast_node_index + batch_tree_index).shape)
        all_node_embeddings = self.hgt_graph_layer(data, graphs = graphs)
        outputs = []
        stmt_lengths = []
        if self.num_node_types == 2:
            graphs.nodes['stmt_node'].data['ft'] = all_node_embeddings['stmt_node']

            for graph in dgl.unbatch(graphs):
                embs = graph.nodes['stmt_node'].data['ft']
                outputs.append(embs)
                stmt_lengths.append(embs.shape[0])
        else:
            graphs.nodes['node'].data['ft'] = all_node_embeddings['node']
            for graph, stmt in zip(dgl.unbatch(graphs), stmt_ids):
                embs = graph.nodes['node'].data['ft']
                outputs.append(embs)
                stmt_lengths.append(embs.shape[0])
        
        memory_bank = nn.utils.rnn.pad_sequence(outputs, batch_first = True, padding_value = 0)
        enc_outputs = memory_bank

        
        stmt_lengths = torch.tensor(stmt_lengths).to(memory_bank.device)
        # tgt_mask, tgt_padding_mask = create_target_mask(label_word_ids, padding_idx = PAD_IDX)
        if self.training:
            tgt_lengths = torch.count_nonzero(label_word_ids != PAD_IDX, dim = 1)
            tgt_initial_embeds = self.tgt_embedder(label_word_ids)
            tgt_pad_mask = ~sequence_mask(tgt_lengths, max_len = tgt_initial_embeds.shape[1])
            # print('label_word_ids', label_word_ids.shape, enc_outputs.shape, tgt_pad_mask.shape)
            layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                    stmt_lengths,
                                                    tgt_pad_mask,
                                                    tgt_initial_embeds
                                                )
            decoder_outputs = layer_wise_dec_out[-1]

            if self.apply_copy:
                src_token_embeds = self.tgt_embedder(src_token_ids_vocab)
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                src_token_embeds,
                                                memory_lengths = src_lengths,
                                                softmax_weights = False)
                attn_copy = F.softmax(copy_score, dim = -1)
                out_generator = self.copy_generator(decoder_outputs, attn_copy, src_map)
                out_generator = out_generator[:, :-1].contiguous()
            else:
                out_generator = self.generator(decoder_outputs[:, :-1]).transpose(1, 2)

            return out_generator
        else:
            kwargs['source_vocab'] = extra_src_info['src_vocabs']
            blank, fill = collapse_copy_scores(self.tgt_dict, extra_src_info['src_vocabs'])
            kwargs['fill'] = fill
            kwargs['blank'] = blank
            src_token_embeds = self.tgt_embedder(src_token_ids_vocab) if self.apply_copy else None
            self.layer_wise_attn = False
            return self.decode(memory_bank, 
                            src_lengths,
                            src_map,
                            alignments,
                            stmt_lengths,
                            src_token_embeds, 
                            **kwargs)
    def decode(self,
            memory_bank,
            code_len,
            src_map,
            alignment,
            stmt_lengths,
            src_token_embeds,
            **kwargs):

        params = dict()
        params['memory_bank'] = memory_bank
        params['layer_wise_outputs'] = None
        params['src_len'] = code_len
        params['stmt_len'] = stmt_lengths
        params['source_vocab'] = kwargs['source_vocab']
        params['src_map'] = src_map
        params['tgt_dict'] = self.tgt_dict
        params['max_len'] = self.max_seq_length
        params['fill'] = kwargs['fill']
        params['src_token_embeds'] = src_token_embeds
        params['blank'] = kwargs['blank']

        dec_preds, attentions, _ = self.__generate_sequence(params, choice='greedy')
        dec_preds = torch.stack(dec_preds, dim=1)
        # attentions: batch_size x tgt_len x num_heads x src_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_preds,
            'memory_bank': memory_bank,
            'attentions': attentions
        }

    def __generate_sequence(self,
                            params,
                            choice='greedy',
                            tgt_words=None):
        src_token_embeds = params['src_token_embeds']
        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([SOS_IDX])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1

        dec_preds = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]
        dec_states = self.decoder.init_decoder(params['stmt_len'], max_mem_len)

        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn \
            else params['memory_bank']
        attns = {"coverage": None}
        # +1 for <EOS> token
        for idx in range(params['max_len']):
            # tgt = self.embedder(tgt_words,
            #                     mode='decoder',
            #                     step=idx)
            tgt = self.tgt_embedder(tgt_words, step = idx)
            tgt_pad_mask = tgt_words.data.eq(PAD_IDX)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            if self.apply_copy:
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                  src_token_embeds,
                                                  memory_lengths=params['src_len'],
                                                  softmax_weights=False)

                attn_copy = F.softmax(copy_score, dim=-1)
                prediction = self.copy_generator(decoder_outputs,
                                                 attn_copy,
                                                 params['src_map'])
                prediction = prediction.squeeze(1)
                for b in range(prediction.size(0)):
                    if params['blank'][b]:
                        blank_b = torch.LongTensor(params['blank'][b])
                        fill_b = torch.LongTensor(params['fill'][b])
                        if use_cuda:
                            blank_b = blank_b.cuda()
                            fill_b = fill_b.cuda()
                        prediction[b].index_add_(0, fill_b,
                                                 prediction[b].index_select(0, blank_b))
                        prediction[b].index_fill_(0, blank_b, 1e-10)

            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = F.softmax(prediction, dim=1)
            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))

            words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])
            words = [params['tgt_dict'].get(w, UNK_IDX) for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt_words = words.unsqueeze(1)

        return dec_preds, attentions, dec_log_probs

    def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict.inverse[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words



class NewMaskedHGTTransformer(nn.Module):
    def __init__(self, config, metadata, tgt_dict, dgl_format, func, num_node_types, *, tree_aggr, pos_encoding, apply_copy):
        super().__init__()
        self.dgl_format = dgl_format
        self.tree_aggr = tree_aggr
        self.max_seq_length = config['max_seq_length']
        self.node_embedding_layer = NodeEmbedding(config)
        self.ast_embedding_layer = ASTNodeEmbeddingFFD(config['out_channels'], self.node_embedding_layer)
        self.tbcnn_layer = Token_TBCNNFFDBlock(self.node_embedding_layer, config, tree_aggr, pos_encoding)
        self.hgt_graph_layer = HGTLayer(config, metadata, dgl_format, func)
        self.num_node_types = num_node_types
        self.pos_encoding = pos_encoding
        self.apply_copy = apply_copy
        self.tgt_dict = tgt_dict

        self.mid_ffd_layer = FeedForward(config['out_channels'])
        self.cross_attention = CrossAttention(config['num_heads_decoder'], config['out_channels'])

        self.stmt_embed_norm = LayerNorm(config['out_channels'])
        self.token_embed_norm = LayerNorm(config['out_channels'])
        self.decoder = Decoder(config)
        self.tgt_embedder = Embedder(config['vocab_comment_size'], config['out_channels'], PAD_IDX)
        self.generator = nn.Linear(config['out_channels'], config['vocab_comment_size'])
        if self.pos_encoding:
            self.central_embedding = nn.Embedding(150, config['out_channels'])
        
        if self.apply_copy:
            self.copy_attn = GlobalAttention(dim = config['out_channels'],
                                            attn_type = config['attn_type'])
            self.copy_generator = CopyGenerator(config['out_channels'], tgt_dict, self.generator)
    def forward(self, num_nodes, ast_node_index, tree, graphs, in_degrees, stmt_ids, extra_src_info, label_word_ids = None, **kwargs):
        ast_node_index, ast_node_embeddings = self.ast_embedding_layer(ast_node_index)
        sizes = list(tree.keys())
        node_embeddings, batch_tree_index, batch_token_node_embeds = [], [], []
        if self.training:
            random.shuffle(sizes)
        for size in sizes:
            batch = tree[size]
            each_bucket_embeddings, bucket_token_node_embeds = self.tbcnn_layer(batch['batch_node_index'], batch['batch_node_type_id'], batch['batch_node_height'], batch['batch_node_sub_tokens_id'], batch['batch_token_node_ids'], batch['batch_children_index'])
            node_embeddings.append(each_bucket_embeddings)
            batch_tree_index.extend(batch['batch_tree_index'])
            _batch_token_node_embeds = []
            for i, token_node_len in enumerate(batch['batch_token_node_lens'].long()):
                _batch_token_node_embeds.append(bucket_token_node_embeds[i][:token_node_len])
            batch_token_node_embeds.extend(_batch_token_node_embeds)
        stmt_node_embeddings = torch.cat(node_embeddings, axis = 0)
        src_map = extra_src_info['src_map']
        src_vocabs = extra_src_info['src_vocabs']
        alignments = extra_src_info['alignments']
        src_token_ids_vocab = extra_src_info['src_token_ids_vocab']
        src_lengths = extra_src_info['src_lengths']
        
        
        # embeddings = torch.cat((ast_node_embeddings, node_embeddi0ngs), dim = 0)
        if self.num_node_types == 1:
            embeddings = torch.cat((ast_node_embeddings, stmt_node_embeddings), dim = 0)

            # ensure that after merging, the order of nodes matches the order in graph
            merge_index = ast_node_index + batch_tree_index
            order_indices = np.argsort(merge_index)
            # print('a', order_indices.shape)

            batch_tokens = [[] for _ in range(len(num_nodes['num_nodes']))]
            batch_mapping_ids = []
            for i, num in enumerate(num_nodes['num_nodes']):
                batch_mapping_ids.extend([i] * num)

            ast_nodes_size = len(ast_node_index)
            for i, index in enumerate(order_indices):
                if merge_index[index] in ast_node_index:
                    batch_tokens[batch_mapping_ids[i]].append(embeddings[index].unsqueeze(0))
                else:
                    j = index - ast_nodes_size
                    batch_tokens[batch_mapping_ids[i]].append(batch_token_node_embeds[j])
            
            batch_token_lens = []
            for i in range(len(batch_tokens)):
                batch_tokens[i] = torch.cat(batch_tokens[i], dim = 0)
                batch_token_lens.append(batch_tokens[i].shape[0])
            
            batch_tokens = nn.utils.rnn.pad_sequence(batch_tokens, batch_first = True)
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
        # print('data', data['node'].shape, np.unique(ast_node_index + batch_tree_index).shape)
        all_node_embeddings = self.hgt_graph_layer(data, graphs = graphs)
        outputs = []
        stmt_lengths = []
        if self.num_node_types == 2:
            graphs.nodes['stmt_node'].data['ft'] = all_node_embeddings['stmt_node']

            for graph in dgl.unbatch(graphs):
                embs = graph.nodes['stmt_node'].data['ft']
                outputs.append(embs)
                stmt_lengths.append(embs.shape[0])
        else:
            graphs.nodes['node'].data['ft'] = all_node_embeddings['node']
            for graph, stmt in zip(dgl.unbatch(graphs), stmt_ids):
                embs = graph.nodes['node'].data['ft']
                outputs.append(embs)
                stmt_lengths.append(embs.shape[0])

        batch_token_lens = torch.tensor(batch_token_lens)
        stmt_lengths = torch.tensor(stmt_lengths)
        memory_bank = nn.utils.rnn.pad_sequence(outputs, batch_first = True, padding_value = 0)

        memory_bank = self.stmt_embed_norm(memory_bank)
        batch_tokens = self.token_embed_norm(batch_tokens)

        fused_two_source_embeds = self.cross_attention(batch_tokens, batch_tokens, memory_bank, batch_token_lens, stmt_lengths)
        _outputs = []
        for fused_emb, stmt_emb, length in zip(fused_two_source_embeds, memory_bank, stmt_lengths):
            _tmp_emb = torch.cat((fused_emb[:length], stmt_emb[:length]), dim = 0)
            _outputs.append(_tmp_emb)
        enc_outputs = nn.utils.rnn.pad_sequence(_outputs, batch_first = True, padding_value = 0)
        # residual_fused_embeds = self.mid_ffd_layer(fused_two_source_embeds + memory_bank)
        # enc_outputs = residual_fused_embeds

        # batch_token_lens = batch_token_lens.to(memory_bank.device)
        stmt_lengths = stmt_lengths.to(memory_bank.device) * 2

        # fused_two_source_embeds = self.cross_attention(memory_bank, memory_bank, batch_tokens, stmt_lengths, batch_token_lens)
        # residual_fused_embeds = self.mid_ffd_layer(fused_two_source_embeds + batch_tokens)
        # enc_outputs = residual_fused_embeds

        # batch_token_lens = batch_token_lens.to(memory_bank.device)
        # tgt_mask, tgt_padding_mask = create_target_mask(label_word_ids, padding_idx = PAD_IDX)
        if self.training:
            tgt_lengths = torch.count_nonzero(label_word_ids != PAD_IDX, dim = 1)

            tgt_initial_embeds = self.tgt_embedder(label_word_ids)
            tgt_pad_mask = ~sequence_mask(tgt_lengths, max_len = tgt_initial_embeds.shape[1])
            # print('label_word_ids', label_word_ids.shape, enc_outputs.shape, tgt_pad_mask.shape)
            layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                    stmt_lengths,
                                                    tgt_pad_mask,
                                                    tgt_initial_embeds
                                                )
            decoder_outputs = layer_wise_dec_out[-1]

            if self.apply_copy:
                # src_token_embeds = self.tgt_embedder(src_token_ids_vocab)
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                enc_outputs,
                                                memory_lengths = stmt_lengths,
                                                softmax_weights = False)
                attn_copy = F.softmax(copy_score, dim = -1)
                out_generator = self.copy_generator(decoder_outputs, attn_copy, src_map)
                out_generator = out_generator[:, :-1].contiguous()
            else:
                out_generator = self.generator(decoder_outputs[:, :-1]).transpose(1, 2)

            return out_generator
        else:
            kwargs['source_vocab'] = extra_src_info['src_vocabs']
            blank, fill = collapse_copy_scores(self.tgt_dict, extra_src_info['src_vocabs'])
            kwargs['fill'] = fill
            kwargs['blank'] = blank
            src_token_embeds = self.tgt_embedder(src_token_ids_vocab) if self.apply_copy else None
            self.layer_wise_attn = False
            return self.decode(enc_outputs, 
                            src_lengths,
                            src_map,
                            alignments,
                            stmt_lengths,
                            src_token_embeds, 
                            **kwargs)
    def decode(self,
            memory_bank,
            code_len,
            src_map,
            alignment,
            stmt_lengths,
            src_token_embeds,
            **kwargs):

        params = dict()
        params['memory_bank'] = memory_bank
        params['layer_wise_outputs'] = None
        params['src_len'] = code_len
        params['stmt_len'] = stmt_lengths
        params['source_vocab'] = kwargs['source_vocab']
        params['src_map'] = src_map
        params['tgt_dict'] = self.tgt_dict
        params['max_len'] = self.max_seq_length
        params['fill'] = kwargs['fill']
        params['src_token_embeds'] = src_token_embeds
        params['blank'] = kwargs['blank']

        dec_preds, attentions, _ = self.__generate_sequence(params, choice='greedy')
        dec_preds = torch.stack(dec_preds, dim=1)
        # attentions: batch_size x tgt_len x num_heads x src_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_preds,
            'memory_bank': memory_bank,
            'attentions': attentions
        }

    def __generate_sequence(self,
                            params,
                            choice='greedy',
                            tgt_words=None):
        src_token_embeds = params['src_token_embeds']
        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([SOS_IDX])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1

        dec_preds = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]
        dec_states = self.decoder.init_decoder(params['stmt_len'], max_mem_len)

        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn \
            else params['memory_bank']
        attns = {"coverage": None}
        # +1 for <EOS> token
        for idx in range(params['max_len'] + 1):
            # tgt = self.embedder(tgt_words,
            #                     mode='decoder',
            #                     step=idx)
            tgt = self.tgt_embedder(tgt_words, step = idx)
            # tgt = self.tgt_embedder(tgt_words)
            tgt_pad_mask = tgt_words.data.eq(PAD_IDX)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            if self.apply_copy:
                _, copy_score, _ = self.copy_attn(decoder_outputs,
                                                  src_token_embeds,
                                                  memory_lengths=params['src_len'],
                                                  softmax_weights=False)

                attn_copy = F.softmax(copy_score, dim=-1)
                prediction = self.copy_generator(decoder_outputs,
                                                 attn_copy,
                                                 params['src_map'])
                prediction = prediction.squeeze(1)
                for b in range(prediction.size(0)):
                    if params['blank'][b]:
                        blank_b = torch.LongTensor(params['blank'][b])
                        fill_b = torch.LongTensor(params['fill'][b])
                        if use_cuda:
                            blank_b = blank_b.cuda()
                            fill_b = fill_b.cuda()
                        prediction[b].index_add_(0, fill_b,
                                                 prediction[b].index_select(0, blank_b))
                        prediction[b].index_fill_(0, blank_b, 1e-10)

            else:
                prediction = self.generator(decoder_outputs.squeeze(1))
                prediction = F.softmax(prediction, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))

            words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])
            words = [params['tgt_dict'].get(w, UNK_IDX) for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt_words = words.unsqueeze(1)

        return dec_preds, attentions, dec_log_probs

    def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict.inverse[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words