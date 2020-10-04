from typing import Any, List

from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models

from da_ssd.model.loss import FocalLoss


class ImageLevelAdaptationHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, features_channels: List[int]):
        super(ImageLevelAdaptationHead, self).__init__()

        self.da_heads = []

        for in_channels in features_channels:
            da_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, 1, kernel_size=1, stride=1)
            )
            self.da_heads.append(da_head)

        for head in self.da_heads:
            for l in head:
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.001)
                    torch.nn.init.constant_(l.bias, 0)

        self.da_heads = nn.ModuleList(self.da_heads)

    def forward(self, x):
        assert len(x) == len(self.da_heads)

        img_features = []
        for feature, head in zip(x, self.da_heads):
            img_features.append(head(feature))

        return img_features


# class ImageLevelAdaptationHead(nn.Module):
#     """
#     Adds a simple Image-level Domain Classifier head
#     """
#
#     def __init__(self, in_channels):
#         super(ImageLevelAdaptationHead, self).__init__()
#
#         self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
#         self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)
#
#         for l in [self.conv1_da, self.conv2_da]:
#             torch.nn.init.normal_(l.weight, std=0.001)
#             torch.nn.init.constant_(l.bias, 0)
#
#     def forward(self, x):
#         img_features = []
#         for feature in x:
#             t = F.relu(self.conv1_da(feature))
#             img_features.append(self.conv2_da(t))
#         return img_features


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class AdaptiveSSDLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, alpha=1, hard_negative_minning=True):
        super(AdaptiveSSDLoss, self).__init__()
        self.alpha = alpha
        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        # you can try to train with both of the loss and look at difference in results over same number of steps
        self.con_loss = FocalLoss(size_average=False)  # nn.CrossEntropyLoss(reduction='none')
        self.hard_negative_minning = hard_negative_minning

    def forward(self, ploc, plabel, gloc, glabel, domain_label):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels
            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        gloc = gloc.transpose(2, 1)
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        source_mask = (domain_label == 0).unsqueeze(1)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, gloc).sum(dim=1)
        sl1 = (source_mask.float() * mask.float() * sl1).sum(dim=1)

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
            closs = (con * (source_mask.float() * mask.float() + neg_mask.float())).sum(dim=1)
        else:
            closs = con.sum(dim=1)

        total_loss = sl1 + self.alpha * closs
        metrics = {'loc': sl1.mean().detach().item(), 'conf': closs.mean().detach().item()}
        # avoid no object detected
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret, metrics

#
# class ImageLevelAdaptationLoss(nn.Module):
#     def forward(self, img_level_features, domain_target):
#         da_img_flattened = []
#         da_img_labels_flattened = []
#         # for each feature level, permute the outputs to make them be in the
#         # same format as the labels. Note that the labels are computed for
#         # all feature levels concatenated, so we keep the same representation
#         # for the image-level domain alignment
#         for da_img_per_level in img_level_features:
#             N, A, H, W = da_img_per_level.shape
#
#             da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
#             da_img_per_level = da_img_per_level.reshape(N, -1)
#
#             da_img_label_per_level = da_img_per_level.new_ones(da_img_per_level.shape)
#             da_img_label_per_level = da_img_label_per_level.reshape(N, -1) * domain_target.unsqueeze(1)
#
#             da_img_flattened.append(da_img_per_level)
#             da_img_labels_flattened.append(da_img_label_per_level)
#
#         da_img_flattened = torch.cat(da_img_flattened, dim=0)
#         da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)
#
#         da_img_loss = F.binary_cross_entropy_with_logits(
#             da_img_flattened, da_img_labels_flattened
#         )
#
#         return da_img_loss
