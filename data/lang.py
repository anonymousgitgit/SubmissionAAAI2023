from bidict import bidict

from constants import *

class Lang:
    def __init__(self, path):
        with open(path) as f:
            lines = list(map(lambda x: x.strip(), f.readlines()))
        self.word2index = bidict(zip(lines, range(len(lines))))
    def get_word_index(self, word):
        return self.word2index.get(word, self.word2index['[OOV]'])
    def get_word_from_index(self, index):
        return self.word2index.inverse[index]
    @property
    def vocab_size(self):
        return len(self.word2index)
    
    def get_words_from_ids(self, lst):
        sentence = []
        for item in lst:
            if item == EOS_IDX: return sentence
            if item < EOS_IDX: continue
            sentence.append(self.word2index.inverse[item])
        return sentence

    def get_sentence_from_ids(self, lst):
        sentence = self.get_words_from_ids(lst)
        if len(sentence) == 0: return ['']
        s = ' '.join(sentence)
        return [s]
        
                
