# src: https://github.com/wasiahmad/NeuralCodeSum/blob/master/c2nl/modules/position_ffn.py
"""
Position feed-forward network from "Attention is All You Need"
"""

import torch.nn as nn
from modules.util_class import LayerNorm

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate = 0.1):
        super().__init__()
        self.layer_norm = LayerNorm(d_model)
        self.relu = nn.ReLU(inplace = True)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        output = self.dropout(self.relu(self.linear(self.layer_norm(x))))
        return output + x


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.intermediate = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.
        Args:
            input: [ batch_size, input_len, model_dim ]
        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.intermediate(self.layer_norm(x))))
        output = self.dropout_2(self.output(inter))
        return output + x