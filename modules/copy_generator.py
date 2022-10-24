# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/copy_generator.py
""" Generator module """
from turtle import hideturtle
import torch.nn as nn
import torch

from constants import PAD_IDX
from modules.utils import aeq

class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.
    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.
    The copy generator is an extended version of the standard
    generator that computes three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary
    """

    def __init__(self, input_size, tgt_dict, generator, eps=1e20):
        super(CopyGenerator, self).__init__()
        self.linear = generator
        self.linear_copy = nn.Linear(input_size, 1)
        self.tgt_dict = tgt_dict
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.eps = eps

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.
        Args:
           hidden (`FloatTensor`): hidden outputs `[batch, tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch, tlen, slen]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[batch, src_len, extra_words]`
        """
        # CHECKS
        batch, tlen, _ = hidden.size()
        batch_, tlen_, slen = attn.size()
        batch, slen_, cvocab = src_map.size()
        aeq(tlen, tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, :, PAD_IDX] = -self.eps
        prob = self.softmax(logits)

        # Probability of copying p(z=1) batch.
        p_copy = self.sigmoid(self.linear_copy(hidden))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy.expand_as(prob))
        mul_attn = torch.mul(attn, p_copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn, src_map)  # `[batch, tlen, extra_words]`
        return torch.cat([out_prob, copy_prob], 2)