import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.supervisedLoss = nn.CrossEntropyLoss()
        
    def forward(self, z, labels, zglob, zprev, tau, mue):
        supervised_loss = self.supervisedLoss(z, labels)
        
        numerator = torch.exp(F.cosine_similarity(z, zglob) / tau)
        denominator = torch.exp(F.cosine_similarity(z, zglob) / tau) + torch.exp(F.cosine_similarity(z, zprev) / tau)
        contrastive_loss = -torch.log(numerator / denominator).mean()
        
        loss = supervised_loss + mue * contrastive_loss
        return loss