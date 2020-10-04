import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        ce = self.ce(input, target)
        pt = (-1 * ce).exp()
        loss = (1 - pt) ** self.gamma * ce
        if self.size_average:
            return loss.mean()
        else:
            return loss


class DAFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(DAFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class SSDLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, alpha=1, hard_negative_minning=True):
        super(SSDLoss, self).__init__()
        self.alpha = alpha
        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        # you can try to train with both of the loss and look at difference in results over same number of steps
        self.con_loss = FocalLoss(size_average=False)  # nn.CrossEntropyLoss(reduction='none')
        self.hard_negative_minning = hard_negative_minning

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels
            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        gloc = gloc.transpose(2, 1)
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, gloc).sum(dim=1)
        sl1 = (mask.float() * sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)
        if self.hard_negative_minning:
            # postive mask will never selected
            con_neg = con.clone()
            con_neg[mask] = 0
            con_neg_top, con_idx = con_neg.sort(dim=1, descending=True)  # aka 8000,1734,2000,...
            _, con_rank = con_idx.sort(dim=1)  # aka 1,2,0,..

            # number of negative three times positive
            neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)  # Nx1

            neg_mask = con_rank < neg_num

            # summing positive confidence and negative confidence
            closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)
        else:
            closs = con.sum(dim=1)

        total_loss = sl1 + self.alpha * closs
        metrics = {'loc': sl1.mean().detach().item(), 'conf': closs.mean().detach().item()}
        # avoid no object detected
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret, metrics
