# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py
""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, d_k, d_v, dropout=0.1, use_neg_dist=True, coverage = False):
        super(MultiHeadedAttention, self).__init__()

        self.head_count = head_count
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v

        self.key = nn.Linear(model_dim, head_count * self.d_k)
        self.query = nn.Linear(model_dim, head_count * self.d_k)
        self.value = nn.Linear(model_dim, head_count * self.d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.head_count * d_v, model_dim)

        self._coverage = coverage

    def forward(self, key, value, query, mask=None, layer_cache=None,
                attn_type=None, step=None, coverage = None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        use_gpu = key.is_cuda

        def shape(x, dim):
            """  projection """
            return x.view(batch_size, -1, head_count, dim).transpose(1, 2)

        def unshape(x, dim):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                # 1) Project key, value, and query.
                key = shape(self.key(key), self.d_k)
                value = shape(self.value(value), self.d_v)
                query = shape(self.query(query), self.d_k)

                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value

            elif attn_type == "context":
                query = shape(self.query(query), self.d_k)
                if layer_cache["memory_keys"] is None:
                    key = shape(self.key(key), self.d_k)
                    value = shape(self.value(value), self.d_v)
                else:
                    key, value = layer_cache["memory_keys"], \
                                 layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = shape(self.key(key), self.d_k)
            value = shape(self.value(value), self.d_v)
            query = shape(self.query(query), self.d_k)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(self.d_k)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        scores = query_key.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)
        
        exp_score = None
        if self._coverage and attn_type == 'context':
        #if 0:
            # batch x num_heads x query_len x 1
            maxes = torch.max(scores, 3, keepdim=True)[0]
            # batch x num_heads x query_len x key_len
            exp_score = torch.exp(scores - maxes)

            if step is not None:  # indicates inference mode (one-step at a time)
                if coverage is None:
                    # t = 1 in Eq(3) from Paulus et al., 2018
                    unnormalized_score = exp_score
                else:
                    # t = otherwise in Eq(3) from Paulus et al., 2018
                    assert coverage.dim() == 4  # B x num_heads x 1 x key_len
                    unnormalized_score = exp_score.div(coverage + 1e-8)
            #if query_len == 1:
            #    unnormalized_score = exp_score
            else:
                multiplier = torch.tril(torch.ones(query_len - 1, query_len - 1))
                # batch x num_heads x query_len-1 x query_len-1
                multiplier = multiplier.unsqueeze(0).unsqueeze(0). \
                    expand(batch_size, head_count, *multiplier.size())
                multiplier = multiplier.cuda() if scores.is_cuda else multiplier

                # B x num_heads x query_len-1 x key_len
                # print('penalty1', exp_score.min(), exp_score.max())
                penalty = torch.matmul(multiplier, exp_score[:, :, :-1, :])
                # B x num_heads x key_len
                no_penalty = torch.ones_like(penalty[:, :, -1, :])
                # B x num_heads x query_len x key_len
                penalty = torch.cat([no_penalty.unsqueeze(2), penalty], dim=2)
                assert exp_score.size() == penalty.size()
                unnormalized_score = exp_score.div(penalty + 1e-8)

            # Eq.(4) from Paulus et al., 2018
            attn = unnormalized_score.div(unnormalized_score.sum(3, keepdim=True))

        # Softmax to normalize attention weights
        else:
            # 3) Apply attention dropout and compute context vectors.
            attn = self.softmax(scores).to(query.dtype)

        # 3) Apply attention dropout and compute context vectors.
        # attn = self.softmax(scores).to(query.dtype)

        # ----------------------------

        # 3) Apply attention dropout and compute context vectors.
        # attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)
        context_original = torch.matmul(drop_attn, value)
        context = unshape(context_original, self.d_v)
        final_output = self.output(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_per_head = [attn.squeeze(1)
                         for attn in attn.chunk(head_count, dim=1)]

        _coverage_vector = None
        if (self._coverage and attn_type == 'context') and step is not None:
            if coverage is None:
                _coverage_vector = exp_score  # B x num_heads x 1 x key_len
            else:
                _coverage_vector = coverage + exp_score
        return final_output, attn_per_head, _coverage_vector

    def update_dropout(self, dropout):
        self.dropout.p = dropout