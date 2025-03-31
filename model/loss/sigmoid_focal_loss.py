import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

# from https://github.com/CoinCheung/pytorch-loss/blob/master/focal_loss.py

class SigmoidFocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = SigmoidFocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        sigmoid(x) = 1/[1+exp(-x)]
        F.softplus(x, beta) = 1/beta * log(1 + exp(beta * x))
        F.softplus(x, -1) = -log([1+exp(-x)]) = log(1/[1+exp(-x)])  = log(sigmoid(x))
        x - F.softplus(x, 1) = x - log([1+exp(x)]) = log(exp(x)/[1+exp(x)]) = log(1/[1+exp(-x)]) = log(sigmoid(x))
        -x + F.softplus(x, -1) = -x - log([1+exp(-x)]) = log(exp(-x)/[1+exp(-x)]) = log(1/[1+exp(x)]) = log(1 - sigmoid(x))
        -F.softplus(x, 1) = -log([1+exp(x)]) = log(1/[1+exp(x)]) = log(1 - sigmoid(x))
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss