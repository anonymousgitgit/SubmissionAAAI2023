from bidict import bidict
from constants import *

class DynamicVocab:
    def __init__(self, no_special_token = False):
        if no_special_token:
            self.tok2id = bidict({
                    '<PAD>': PAD_IDX,
                    '<UNK>': UNK_IDX
                })
        else:
            self.tok2id = bidict({
                    '<PAD>': PAD_IDX,
                    '<UNK>': UNK_IDX,
                    '<SOS>': SOS_IDX,
                    '<EOS>': EOS_IDX
            })
    def __len__(self):
        return len(self.tok2id)
    def __iter__(self):
        return iter(self.tok2id)
    def add_tokens(self, tokens):
        for token in tokens:
            self.add(token)
    def add(self, token):
        if token not in self.tok2id:
            index = len(self.tok2id)
            self.tok2id[token] = index
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.tok2id.get(key, self.tok2id['<UNK>'])
        elif isinstance(key, int):
            try:
                return self.tok2id.inverse[key]
            except:
                return self.tok2id.inverse['<UNK>']
        else:
            raise RuntimeError("Invalid key type")
    def __setitem__(self, key, item):
        if isinstance(key, str) and isinstance(item, int):
            self.tok2id[key] = item
        else:
            raise RuntimeError("Invalid (key, item) types")
    