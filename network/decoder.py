from turtle import end_fill
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class RuleEmbedding(nn.Module):
    def __init__(self, rule_size, rule_token_size, dim_size):
        super().__init__()
        self.w = nn.Linear(dim_size * 3, dim_size)
        self.rule_embeds = nn.Embedding(rule_size + 1, dim_size, padding_idx = 0)
        self.token_embeds = nn.Embedding(rule_token_size + 1, dim_size, padding_idx = 0)
    def forward(self, rule_token_ids, rule_ids):
        # rules: N x S x L
        # rule embeds: N x S x L x D
        rule_token_embeds = self.token_embeds(rule_token_ids)
        rule_id_embeds = self.rule_embeds(rule_ids)
        # N x S x D
        parent_embeds = rule_token_embeds[..., 0, :]
        # N x S x D
        content_embeds = torch.sum(rule_token_embeds, axis = 2)
        r_embeds = torch.relu(torch.cat((parent_embeds, content_embeds, rule_id_embeds), dim = -1))
        return self.w(r_embeds) 

class RuleEmbedder(nn.Module):
    def __init__(self, rule_size, rule_token_size, dim_size, sos_idx, eos_idx):
        super().__init__()
        self.sos_idx = torch.tensor(sos_idx)
        self.eos_idx = torch.tensor(eos_idx)
        self.rule_embedding = RuleEmbedding(rule_size, rule_token_size, dim_size)
        self.pos_enc_layer = PositionalEncoding(0.1, dim_size)
        self.pos_embedding = nn.Embedding(100, dim_size)
    def forward(self, rule_token_ids, rule_ids, step = None):
        if self.training:
            # sos_embedding = self.rule_embedding.rule_embeds(self.sos_idx.to(rule_ids.device))
            # eos_embedding = self.rule_embedding.rule_embeds(self.eos_idx.to(rule_ids.device))
            # sos_embedding = sos_embedding.unsqueeze(0).unsqueeze(0)
            # eos_embedding = eos_embedding.unsqueeze(0).unsqueeze(0)
            
            rule_rep = self.rule_embedding(rule_token_ids.long(), rule_ids.long())
            batch_size = rule_rep.shape[0]
            rule_rep = self.pos_enc_layer(rule_rep)
            if step is None:
                pos_enc = torch.arange(start=0,
                                        end=rule_rep.size(1)).type(torch.LongTensor)
            else:
                pos_enc = torch.LongTensor([step])  # used in inference time

            pos_enc = pos_enc.expand(*rule_rep.size()[:-1]).to(rule_rep.device)
            pos_rep = self.pos_embedding(pos_enc)
            rule_rep = rule_rep + pos_rep
        else:
            rule_rep = self.rule_embedding(rule_token_ids.long(), rule_ids.long())
            rule_rep = self.pos_enc_layer(rule_rep)
            pos_enc = torch.LongTensor([step])  # used in inference time

            pos_enc = pos_enc.expand(*rule_rep.size()[:-1]).to(rule_rep.device)
            pos_rep = self.pos_embedding(pos_enc)
            rule_rep = rule_rep + pos_rep

        return rule_rep


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim))).unsqueeze(0)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:, :emb.size(1)]
        else:
            emb = emb + self.pe[:, step]
        emb = self.dropout(emb)
        return emb
class Embedder(nn.Module):
    def __init__(self, vocab_comment_size, dim_size, padding_idx = None):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_comment_size, dim_size, padding_idx = padding_idx)
        self.pos_embedding = nn.Embedding(100, dim_size)
        self.pos_enc_layer =  PositionalEncoding(0.1, dim_size)
    def forward(self, word_ids, step = None):
        word_rep = self.word_embedding(word_ids)
        word_rep = self.pos_enc_layer(word_rep)
        if step is None:
            pos_enc = torch.arange(start=0,
                                    end=word_rep.size(1)).type(torch.LongTensor)
        else:
            pos_enc = torch.LongTensor([step])  # used in inference time

        pos_enc = pos_enc.expand(*word_rep.size()[:-1]).to(word_rep.device)
        pos_rep = self.pos_embedding(pos_enc)
        word_rep = word_rep + pos_rep
        return word_rep
# class Embedder(nn.Module):
#     def __init__(self, vocab_comment_size, dim_size, padding_idx = None):
#         super().__init__()
#         self.word_embedding = nn.Embedding(vocab_comment_size, dim_size, padding_idx = padding_idx)
#     def forward(self, word_ids):
#         return self.word_embedding(word_ids)
class DecoderRNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = self.input_size * 2

        self.embedding = nn.Embedding(output_size, input_size)
        self.gru = nn.GRU(input_size, self.hidden_size, batch_first = True)
        self.out = nn.Linear(self.hidden_size, output_size)
    def forward(self, input, h):
        output = self.embedding(input).unsqueeze(1)
        output = torch.relu(output)
        output, h = self.gru(output, h)
        output = self.out(output)
        return output, h

class Attention(nn.Module):
    def __init__(self, method, hidden_dim):
        super().__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if self.method == 'general':
            self.fc = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, hidden_state, encoder_outputs, encoder_mask):
        # batch_size, hidden_dim = hidden_state.shape
        # _, seq_len, _ = encoder_outputs.shape
        if self.method == 'dot':
            scores = torch.einsum('dbh, blh -> bl', hidden_state, encoder_outputs)
        elif self.method == 'general':
            encoder_outputs = self.fc(encoder_outputs)
            scores = torch.einsum('dbh, blh -> bl', hidden_state, encoder_outputs)
        scores = scores + encoder_mask
        scores = F.softmax(scores, dim = 1)
        return scores
        

class AttnDecoderRNN(nn.Module):
    def __init__(self, method, input_size, output_size, dropout_p = 0.1):
        super().__init__()
        self.hidden_size = input_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, input_size)
        self.gru = nn.GRU(input_size, input_size, batch_first = True)
        self.generator = nn.Linear(input_size * 2, output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn_layer = Attention(method, input_size)
        self.mlp = nn.Sequential(
            nn.Linear(input_size * 2, input_size * 2),
            nn.ReLU(inplace = True)
        )
        
    def forward(self, input, decoder_hidden, encoder_outputs, encoder_mask):
        
        embeds = self.embedding(input).unsqueeze(1)
        embeds = torch.relu(embeds)

        output, h = self.gru(embeds, decoder_hidden)
        scores = self.attn_layer(h, encoder_outputs, encoder_mask)
        # scores = self.dropout(scores)
        context = torch.einsum('bl, bld -> bd', scores, encoder_outputs)
        cat_embs = torch.cat((h.squeeze(0), context), dim = -1)
        output = self.mlp(cat_embs)

        # output, new_hidden = self.gru(embeds, torch.cat((decoder_hidden, context), dim = 1).unsqueeze(0))
        output = self.generator(output).unsqueeze(1)
        # new_hidden = self.combine_layer(new_hidden)
        return output, h
