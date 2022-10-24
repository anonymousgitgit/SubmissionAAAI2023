import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import dgl
import random

from network.graph_layers import *
from network.decoder import Embedder
from constants import *
from modules.copy_generator import CopyGenerator
from modules.position_ffn import FeedForward
from modules.util_class import LayerNorm
from modules.utils import sequence_mask, collapse_copy_scores
from network.transformer import TransformerDecoder
from constants.vocab import *
from modules.global_attention import GlobalAttention
from modules.cross_attention import CrossAttention

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



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
        self.lsm = nn.LogSoftmax(dim=-1)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=config['out_channels'], dim_feedforward = config['feedforward_decoder_channels'], nhead = config['num_heads_decoder'], dropout = config['dropout_decoder'], batch_first = True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = config['num_decoder_layers'])
        self.tgt_embedder = Embedder(config['vocab_comment_size'], config['out_channels'], PAD_IDX)
        self.generator = nn.Linear(config['out_channels'], config['vocab_comment_size'])

        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
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
        src_padding_mask = []
        if self.num_node_types == 2:
            graphs.nodes['stmt_node'].data['ft'] = all_node_embeddings['stmt_node']

            for graph in dgl.unbatch(graphs):
                embs = graph.nodes['stmt_node'].data['ft']
                outputs.append(embs)
                stmt_lengths.append(embs.shape[0])
                src_padding_mask.append(torch.tensor([0] * embs.shape[0]).unsqueeze(-1))
        else:
            graphs.nodes['node'].data['ft'] = all_node_embeddings['node']
            for graph, stmt in zip(dgl.unbatch(graphs), stmt_ids):
                embs = graph.nodes['node'].data['ft']
                outputs.append(embs)
                stmt_lengths.append(embs.shape[0])
                src_padding_mask.append(torch.tensor([0] * embs.shape[0]).unsqueeze(-1))
        
        src_padding_mask = nn.utils.rnn.pad_sequence(src_padding_mask, batch_first = True, padding_value = 1).squeeze(-1)
        memory_bank = nn.utils.rnn.pad_sequence(outputs, batch_first = True, padding_value = 0)
        enc_outputs = memory_bank
        src_padding_mask = src_padding_mask.to(memory_bank.device)
        
        stmt_lengths = torch.tensor(stmt_lengths).to(memory_bank.device)
        # tgt_mask, tgt_padding_mask = create_target_mask(label_word_ids, padding_idx = PAD_IDX)
        if self.training:
            # tgt_lengths = torch.count_nonzero(label_word_ids != PAD_IDX, dim = 1)
            tgt_initial_embeds = self.tgt_embedder(label_word_ids)
            tgt_mask = generate_square_subsequent_mask(tgt_initial_embeds.shape[1]).to(tgt_initial_embeds.device)
            
            decoder_outputs = self.decoder(tgt_initial_embeds, enc_outputs,
                                                    tgt_mask = tgt_mask,
                                                    memory_key_padding_mask = src_padding_mask
                                                )
            

    
            out_generator = self.generator(decoder_outputs[:, :-1]).transpose(1, 2)

            return out_generator
        else:
            preds=[]       
            zero=torch.LongTensor(1).fill_(0).to(enc_outputs.device)     
            for i in range(enc_outputs.shape[0]):
                context=enc_outputs[i:i+1]
                context_mask=src_padding_mask[i:i+1,:]
                beam = Beam(self.beam_size,BOS_IDX, EOS_IDX)
                input_ids=beam.getCurrentState()
                context=context.repeat(1, self.beam_size,1)
                context_mask=context_mask.repeat(self.beam_size,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[0],:input_ids.shape[0]])
                    tgt_embeddings = self.tgt_embedder(input_ids)
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(context_mask).bool())
                    out=self.generator(out)
                    out = self.lsm(out).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)                
            return preds   


class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        


