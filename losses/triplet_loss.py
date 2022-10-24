import torch
import torch.nn as nn
class TripletLosses(nn.Module):
    def __init__(self, alpha, lambda1, lambda2):
        super().__init__()
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, anchor_logits, anchor_lbs, anchor_embeds, pos_embeds, neg_embeds):
        ce_loss = self.cross_entropy(anchor_logits, anchor_lbs)
        dot_p = torch.einsum('bi, bi -> b', anchor_embeds, pos_embeds)
        dot_n = torch.einsum('bi, bi -> b', anchor_embeds, neg_embeds)
        mag_a = torch.norm(anchor_embeds, dim = -1)
        mag_p = torch.norm(pos_embeds, dim = -1)
        mag_n = torch.norm(neg_embeds, dim = -1)
        D_plus = 1 - torch.abs(dot_p) / (mag_a * mag_p)
        D_minus = 1 - torch.abs(dot_n) / (mag_a * mag_n)
        con_loss = torch.relu(D_plus - D_minus + self.alpha).mean()
        l2_loss = (mag_a + mag_p + mag_n).mean()

        return ce_loss + self.lambda1 * con_loss + self.lambda2 * l2_loss



