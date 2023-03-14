
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

def kl_div(p_out, q_out, get_softmax=True):
    KLD = nn.KLDivLoss(reduction='batchmean')
    B = p_out.size(0)

    if get_softmax:
        p_out = F.softmax(p_out.view(B,-1),dim=-1)
        q_out = F.log_softmax(q_out.view(B,-1),dim=-1)

    kl_loss = KLD(q_out, p_out)

    return kl_loss

class HM_Loss(nn.Module):
    def __init__(self):
        super(HM_Loss, self).__init__()
        self.gamma = 2
        self.alpha = 0.25

    def forward(self, pred, target):
        #[B, N, 18]
        temp1 = -(1-self.alpha)*torch.mul(pred**self.gamma,
                           torch.mul(1-target, torch.log(1-pred+1e-6)))
        temp2 = -self.alpha*torch.mul((1-pred)**self.gamma,
                           torch.mul(target, torch.log(pred+1e-6)))
        temp = temp1+temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))

        intersection_positive = torch.sum(pred*target, 1)
        cardinality_positive = torch.sum(torch.abs(pred)+torch.abs(target), 1)
        dice_positive = (intersection_positive+1e-6) / \
            (cardinality_positive+1e-6)

        intersection_negative = torch.sum((1.-pred)*(1.-target), 1)
        cardinality_negative = torch.sum(
            2-torch.abs(pred)-torch.abs(target), 1)
        dice_negative = (intersection_negative+1e-6) / \
            (cardinality_negative+1e-6)
        temp3 = torch.mean(1.5-dice_positive-dice_negative, 0)

        DICELoss = torch.sum(temp3)
        return CELoss+1.0*DICELoss

class CrossModalCenterLoss(nn.Module):
    """Center loss.    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim=512, local_rank=None):
        super(CrossModalCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.local_rank = local_rank

        if self.local_rank != None:
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.device = torch.device('cuda:0')
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        temp = torch.mm(x, self.centers.t())
        distmat = distmat - 2*temp

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss



