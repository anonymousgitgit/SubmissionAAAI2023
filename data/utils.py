import re
import os
import unicodedata
import json
import torch
from constants import PAD_IDX


def get_ids_from_file(path):
    indices = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            indices.append(int(line))
    return indices

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.strip())
    # s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"([.])", r" . ", s)
    s = re.sub(r"([?])", r" ? ", s)
    s = re.sub(r"([!])", r" ! ", s)
    s = re.sub(r"([@])", r" @ ", s)
    s = re.sub(r"([%])", r" % ", s)
    s = re.sub(r"([&])", r" & ", s)
    s = re.sub(r"([(])", r" ( ", s)
    s = re.sub(r"([)])", r" ) ", s)
    s = re.sub(r"([;])", r" ; ", s)
    s = re.sub(r"([,])", r" , ", s)
    s = re.sub(r"([#])", r" # ", s)
    s = re.sub(r"([\"])", r" \" ", s)
    s = re.sub(r"[^\"a-zA-Z.!?0-9@%#&();,]+", r" ", s)
    return s

def read_file(path):
    with open(path) as f:
        lines = f.readlines()
    text = ''.join(lines)
    return text.encode("ascii", errors="ignore").decode()

def is_class(path, classes):
    return int(path.split(os.path.sep)[-2]) in classes

def read_and_filter(path, max_length):
    with open(path) as f:
        lines = f.readlines()
    results = []
    for line in lines:
        json_data = json.loads(line)
        comment = json_data['comment']
        processed_comment = normalizeString(comment)
        if len(processed_comment.split()) < max_length:
            json_data['comment'] = processed_comment
            results.append(json_data)
    return results

def generate_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).t()
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.)
    return mask

def create_target_mask(tgt, padding_idx):
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_subsequent_mask(tgt_seq_len).to(tgt.device)

    tgt_padding_mask = tgt == padding_idx

    return tgt_mask, tgt_padding_mask