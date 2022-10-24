class Vocab:
    def __init__(self, vocab_tokens_path, vocab_type_path):
        with open(vocab_tokens_path) as f:
            tokens = list(map(lambda x: x.strip(), f.readlines()))
        with open(vocab_type_path) as f:
            types = list(map(lambda x: x.strip().lower(), f.readlines()))
        self.vocab_tokens_lookup = dict(zip(tokens, range(len(tokens))))
        self.vocab_types_lookup = dict(zip(types, range(len(types))))
        # self.vocab_tokens = set()
        self.vocab_types = set()
    @property
    def vocab_token_size(self):
        return len(self.vocab_tokens_lookup)
    @property
    def vocab_type_size(self):
        return len(self.vocab_types_lookup)
    def get_id_from_node_type(self, node_type):
        node_type = node_type.strip().lower()
        return self.vocab_types_lookup.get(node_type, self.vocab_types_lookup['unk_type']) + 1
    def get_id_from_sub_token(self, token):
        token = token.strip()

        return (self.vocab_tokens_lookup[token] if token in self.vocab_tokens_lookup else self.vocab_tokens_lookup['[<special>]']) + 1