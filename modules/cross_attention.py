import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Reference: https://github.com/sooftware/attentions/blob/master/attentions.py
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        # if mask is not None:
        #     score.masked_fill_(mask.view(score.size()), -float('Inf'))
        score = score + mask

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class CrossAttention(nn.Module):
    def __init__(self, num_heads, dim, dropout_rate = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dropout_rate = dropout_rate

        self.d_head = dim // num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.key_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self, key, value, query, key_lens, query_lens):
        batch_size = key.size(0)
        max_key_length  = key.shape[1]
        max_query_length = query.shape[1]
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD
        
        mask = self.create_mask(key_lens, query_lens, max_key_length, max_query_length).to(key.device)
        
        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context
    def create_mask(self, key_lens, query_lens, max_key_length, max_query_length):
        key_mask = torch.arange(max_key_length).unsqueeze(0) < key_lens.unsqueeze(1)
        query_mask = torch.arange(max_query_length).unsqueeze(0) < query_lens.unsqueeze(1)
        key_mask = key_mask.unsqueeze(1).float()
        query_mask = query_mask.unsqueeze(-1).float()
        mask = key_mask + query_mask
        mask = mask.masked_fill(mask < 2, -1e18)
        mask = mask.masked_fill(mask == 2, 0)
        mask = mask.unsqueeze(0).tile(self.num_heads, 1, 1, 1).view(-1, max_query_length, max_key_length)
        return mask



