import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import dgl
import random

from network.graph_layers import *
from network.decoder import Embedder, RuleEmbedder

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

class MaskedRuleHGTTransformer(nn.Module):
    def __init__(self, config, metadata, tgt_dict, dgl_format, func, num_node_types, *, tree_aggr, pos_encoding, apply_copy, rule_obj = None, rule_token_obj = None):
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
        self.rule_embedder = RuleEmbedder(config['rule_vocab_size'], config['rule_tokens_size'], config['out_channels'], SOS_IDX, EOS_IDX)
        self.generator = nn.Linear(config['out_channels'], config['rule_vocab_size'])
        if self.pos_encoding:
            self.central_embedding = nn.Embedding(150, config['out_channels'])
        
        if self.apply_copy:
            self.copy_attn = GlobalAttention(dim = config['out_channels'],
                                            attn_type = config['attn_type'])
            self.copy_generator = CopyGenerator(config['out_channels'], tgt_dict, self.generator)
        self.rule_obj = rule_obj
        self.rule_token_obj = rule_token_obj
    def forward(self, num_nodes, ast_node_index, tree, graphs, in_degrees, stmt_ids, extra_src_info, rule_token_ids = None, rule_length = None, rule_ids = None, **kwargs):
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
            # tgt_lengths = torch.count_nonzero(label_word_ids != PAD_IDX, dim = 1)

            tgt_initial_embeds = self.rule_embedder(rule_token_ids, rule_ids)
            tgt_pad_mask = ~sequence_mask(rule_length, max_len = tgt_initial_embeds.shape[1])

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

        dec_preds = self.generate_rule(params)

        # dec_preds, attentions, _ = self.__generate_sequence(params, choice='greedy')
        # dec_preds = torch.stack(dec_preds, dim=1)
        # # attentions: batch_size x tgt_len x num_heads x src_len
        # attentions = torch.stack(attentions, dim=1) if attentions else None

        # return {
        #     'predictions': dec_preds,
        #     'memory_bank': memory_bank,
        #     'attentions': attentions
        # }
        return {
            'predictions': dec_preds
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
            tgt_token_ids = torch.LongTensor([[SOS_IDX, SOS_IDX, SOS_IDX]])
            if use_cuda:
                tgt_words = tgt_words.cuda()
                tgt_token_ids = tgt_token_ids.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1
            tgt_token_ids = tgt_token_ids.tile((batch_size, 1))
            


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
            tgt = self.rule_embedder(tgt_token_ids, tgt_words, step = idx)
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

            # words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])
            # words = [params['tgt_dict'].get(w, UNK_IDX) for w in words]
            # words = torch.Tensor(words).type_as(tgt)
            tgt_words = tgt
            tgt_token_ids = torch.Tensor(self.tens2rule(tgt)).type_as(tgt)
        return dec_preds, attentions, dec_log_probs
    def tens2rule(self, tgts):
        rule_token_ids = []
        for tgt in tgts:
            tgt_idx = tgt.item()
            tgt_rule = self.rule_obj.get_word_from_index(tgt_idx)
            if tgt_rule == '[EOS]':
                tgt_tokens = ['[EOS]', '[EOS]', '[EOS]']
            elif tgt_rule == '[SOS]':
                tgt_tokens = ['[SOS]', '[SOS]', '[SOS]']
            elif tgt_rule == '[OOV]':
                tgt_tokens = ['[OOV]', '[OOV]', '[OOV]']
            elif "character_literal : ' '" in tgt_rule:
                tgt_tokens = ['character_literal', ':', "' '"]
            else:
                tgt_tokens = list(map(lambda x: x.strip(), tgt_rule.split()))
            tgt_token_ids = list(map(lambda x: self.rule_token_obj.get_word_index(x), tgt_tokens))
            rule_token_ids.append(tgt_token_ids)
        return rule_token_ids
    def generate_rule(self, params):
        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        tgt_words = torch.LongTensor([SOS_IDX])
        tgt_token_ids = torch.LongTensor([[SOS_IDX, SOS_IDX, SOS_IDX]])
        if use_cuda:
            tgt_words = tgt_words.cuda()
            tgt_token_ids = tgt_token_ids.cuda()
        tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1
        tgt_token_ids = tgt_token_ids.tile((batch_size, 1)).unsqueeze(1)

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]
        dec_states = self.decoder.init_decoder(params['stmt_len'], max_mem_len)

        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn \
            else params['memory_bank']
        stmt_lengths = params['stmt_len']
        for idx in range(params['max_len']):
            tgt = self.rule_embedder(tgt_token_ids, tgt_words, step = idx)
            tgt_pad_mask = tgt_words.data.eq(PAD_IDX)
            layer_wise_dec_out, attns = self.decoder(enc_outputs,
                                                    stmt_lengths,
                                                    tgt_pad_mask,
                                                    tgt
                                                )
            decoder_outputs = layer_wise_dec_out[-1]
            out_generator = self.generator(decoder_outputs[:, -1])
            tgt_idx = torch.argmax(out_generator, dim = 1)
            # print('tgt_idx', tgt_idx.shape)
            _tgt_token_ids = (self.tens2rule(tgt_idx.unsqueeze(1)))
            # print('token ids', _tgt_token_ids)
            _tgt_token_ids = torch.Tensor(_tgt_token_ids).type_as(tgt_idx)
            _tgt_token_ids = _tgt_token_ids.unsqueeze(1)
            if tgt_token_ids is None:
                tgt_token_ids = _tgt_token_ids
            else:
                tgt_token_ids = torch.cat((tgt_token_ids, _tgt_token_ids), dim = 1)
            tgt_words = torch.cat((tgt_words, tgt_idx.unsqueeze(1)), dim = 1)
            # print('tgt_words', tgt_words.shape, tgt_token_ids.shape)
        return tgt_words[:, 1:]

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
