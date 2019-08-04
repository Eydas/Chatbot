import torch

def masked_nll_loss(softmax_probs, targets, mask):
    target_probs = torch.gather(softmax_probs, 1, targets.view(-1, 1)).squeeze(1)
    crossEntropy = -torch.log(target_probs)
    loss = crossEntropy.masked_select(mask).mean()
    return loss, mask.sum().item()