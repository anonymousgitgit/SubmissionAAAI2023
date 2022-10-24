import torch
from constants import *

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

def collapse_copy_scores(tgt_dict, src_vocabs):
    """
    Given scores from an expanded dictionary
    corresponding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_dict)
    blank_arr, fill_arr = [], []
    for b in range(len(src_vocabs)):
        blank = []
        fill = []
        src_vocab = src_vocabs[b]
        # Starting from 2 to ignore PAD and UNK token
        for i in range(2, len(src_vocab)):
            sw = src_vocab[i]
            ti = tgt_dict.get(sw.strip(), UNK_IDX)
            if ti != UNK_IDX:
                blank.append(offset + i)
                fill.append(ti)

        blank_arr.append(blank)
        fill_arr.append(fill)

    return blank_arr, fill_arr

def tens2sen(t, word_dict=None, src_vocabs=None):
    sentences = []
    raw_words = []
    # loop over the batch elements
    for idx, s in enumerate(t):
        sentence = []
        words = []
        for wt in s:
            word = wt if isinstance(wt, int) \
                else wt.item()
            if word in [SOS_IDX]:
                continue
            if word in [EOS_IDX]:
                break
            if word_dict and word < len(word_dict):
                token = word_dict.inverse[word]
                sentence += [token]
            elif src_vocabs:
                word = word - len(word_dict)
                sentence += [src_vocabs[idx][word]]
            else:
                sentence += [str(word)]

        if len(sentence) == 0:
            # NOTE: just a trick not to score empty sentence
            # this has no consequence
            sentence = [str('[PAD]')]

        words = sentence
        sentence = ' '.join(sentence)
        # if not validate(sentence):
        #     sentence = str(constants.PAD)
        raw_words.append(words)
        sentences.append([sentence])
    return sentences, raw_words

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    :param lengths: 1d tensor [batch_size]
    :param max_len: int
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)  # (0 for pad positions)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def make_src_map(data):
    """ ? """
    # src_size = max([t.size(0) for t in data])
    src_size = max([len(t) for t in data])
    # src_vocab_size = max([t.max() for t in data]) + 1
    src_vocab_size = max([max(t) for t in data]) + 1
    alignment = torch.zeros(len(data), src_size, src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[i, j, t] = 1
    return alignment


def align(data):
    """ ? """
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(len(data), tgt_size).long()
    for i, sent in enumerate(data):
        alignment[i, :sent.size(0)] = sent
    return alignment

def replace_unknown(prediction, attn, src_raw):
    """ ?
        attn: tgt_len x src_len
    """
    tokens = prediction.split()
    for i in range(len(tokens)):
        if tokens[i] == '[OOV]':
            _, max_index = attn[i].max(0)
            tokens[i] = src_raw[max_index.item()]
    return ' '.join(tokens)
