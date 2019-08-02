import torch

def masked_nll_loss(softmax_probs, targets, mask):
    target_probs = torch.gather(torch.transpose(softmax_probs, 0, 1), 2, targets.unsqueeze(2)).squeeze(2)
    crossEntropy = -torch.log(target_probs)
    loss = crossEntropy.masked_select(mask).mean()
    return loss