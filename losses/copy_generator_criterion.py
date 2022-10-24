from modules.utils import aeq
from constants import *

class CopyGeneratorCriterion(object):
    """ Copy generator criterion """

    def __init__(self, vocab_size, force_copy, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size

    def __call__(self, scores, align, target):
        # CHECKS
        batch, tlen, _ = scores.size()
        _, _tlen = target.size()
        aeq(tlen, _tlen)
        _, _tlen = align.size()
        aeq(tlen, _tlen)

        align = align.view(-1)
        _target = target
        target = target.view(-1)
        
        scores = scores.view(-1, scores.size(2))

        # Compute unks in align and target for readability
        align_unk = align.eq(UNK_IDX).float()
        align_not_unk = align.ne(UNK_IDX).float()
        target_unk = target.eq(UNK_IDX).float()
        target_not_unk = target.ne(UNK_IDX).float()

        # Copy probability of tokens in source
        out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            out = out + tmp.mul(target_not_unk)
            # Add score for when word is unk in both align and tgt
            out = out + tmp.mul(align_unk).mul(target_unk)
        else:
            # Forced copy. Add only probability for not-copied tokens
            out = out + tmp.mul(align_unk)

        ml_loss = -out.log()
        ml_loss = ml_loss.view(batch, tlen)
        ml_loss = ml_loss.mul(_target.ne(PAD_IDX).float())
        ml_loss = ml_loss.sum(1) 
        return ml_loss.mean()