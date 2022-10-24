import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import os
import numpy as np
import json
import dgl
import torch
import re
import jsonlines
from torch.utils.data import Dataset

from data.mask_convert_java import convert_into_java_graph
from data.tokenizer import split_identifier_into_parts
from data.dynamic_vocab import DynamicVocab

class SLMRNNDataset(Dataset):
    def __init__(self, input_path, output_path, lang_obj, max_seq_length, tokenizer):
        self.input_path = input_path
        self.output_path = output_path
        self.lang_obj = lang_obj
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

        input_data = []
        with open(input_path) as f:
            for line in f:
                input_data.append(line.strip())
        self.input_data = input_data

        output_data = []
        with open(output_path) as f:
            for line in f:
                output_data.append(line.strip())
        self.output_data = output_data
        assert len(input_data) == len(output_data)
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, index):
        input = self.remove_str(self.input_data[index])
        output = self.remove_str(self.output_data[index])
        input_enc = self.tokenizer.encode(input)
        output_enc = self.tokenizer.encode(output)
        in_tokens = ['[SOS]'] + input_enc.tokens + ['[EOS]']
        out_tokens = ['[SOS]'] + output_enc.tokens + ['[EOS]']
        in_token_ids = self.get_ids_from_tokens(in_tokens)
        out_token_ids = self.get_ids_from_tokens(out_tokens)
        return in_token_ids, out_token_ids, output
    def get_ids_from_tokens(self, tokens):
        return list(map(lambda x: self.lang_obj.get_word_index(x), tokens))
    def remove_str(self, text):
        text = re.sub(r'"[\s\S]*?"', ' "str" ', text)
        text = re.sub(r'/\*[\s\S]*?\*/', '', text)
        text = re.sub(r'//.*?\n', '', text)
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        text = text.replace("PRED", "[MASK]")
        return text
