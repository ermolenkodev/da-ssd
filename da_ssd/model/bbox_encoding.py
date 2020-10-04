from math import sqrt
import itertools
import torch
from SSD import _C as C


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, aspect_ratios,
                 scale_xy=0.1, scale_wh=0.2, scale_min=0.07, scale_max=0.9):

        self.feat_size = feat_size
        self.fig_size = fig_size
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        self.aspect_ratios = aspect_ratios

        self.default_boxes = []

        for idx, sfeat in enumerate(self.feat_size):
            sk = scale_min + ((scale_max - scale_min) / (len(self.feat_size) - 1)) * (idx)
            sk_next = scale_min + ((scale_max - scale_min) / (len(self.feat_size) - 1)) * (idx + 1)

            all_sizes = []
            for alpha in self.aspect_ratios[idx]:
                w, h = sk * sqrt(alpha), sk / sqrt(alpha)
                all_sizes.append((w, h))
            # for aspect 1 adding additional box
            # FIXED
            # here was a typo, now sk calculating by formula, as it should
            sk = sqrt(sk * sk_next)
            w, h = sk * sqrt(1.), sk / sqrt(1.)
            all_sizes.append((w, h))

            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / sfeat, (i + 0.5) / sfeat
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes)
        # make values in interval [0,1]
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb.clamp_(min=0, max=1)
        # FIXED
        # LTRB boxes added clamp, cause new values can be out of interval [0,1]

    @property
    def scale_xy(self):
        # also called variance
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes


def default_boxes_300():
    # we use default values
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    aspect_ratios = [[1.,2.,0.5], [1.,2.,0.5,3.,1./3], [1.,2.,0.5,3.,1./3], [1.,2.,0.5,3.,1./3], [1.,2.,0.5], [1.,2.,0.5]]
    scale_min=0.07
    scale_max=0.9
    dboxes = DefaultBoxes(figsize, feat_size, aspect_ratios, scale_min=scale_min, scale_max=scale_max)
    return dboxes


def calc_iou_tensor(box1, box2):
    """ param: box1  tensor (N, 4) (x1,y1,x2,y2)
        param: box2  tensor (M, 4) (x1,y1,x2,y2)
        output: tensor (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    box1 = box1.unsqueeze(1).expand(-1, M, -1)
    box2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(box1[:, :, :2], box2[:, :, :2])
    rb = torch.min(box1[:, :, 2:], box2[:, :, 2:])

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:, :, 0] * delta[:, :, 1]

    delta1 = box1[:, :, 2:] - box1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = box2[:, :, 2:] - box2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = intersect / (area1 + area2 - intersect)
    return iou


class BoxUtils:
    """
        Util to encode/decode target/result boxes

        param: dboxes - DefaultBoxes instance
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh")
        self.nboxes = self.dboxes.size(0)

        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes, labels_in, criteria=0.5, num_classes=21):
        ious = calc_iou_tensor(bboxes, self.dboxes)  # [N_bboxes, num_default_boxes]
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)  # [num_default_boxes], [num_default_boxes]
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)  # [N_bboxes], [N_bboxes]

        # set best ious 2.0
        # this needed to not filter out this bboxes on next step
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        # filter out dboxes with IoU <= criteria
        masks = best_dbox_ious > criteria
        # setting all labels of filtered out dboxes to zero(background)
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]

        # setting ghound trouth boxes on place of best matched default boxes
        # and convert (x1,y1,x2,y2) format to (xc,yc,w,h)
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes[best_dbox_idx[masks], :]
        # Transform format to xywh format
        # x, y, w, h = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2]), \
        #              0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3]), \
        #              -bboxes_out[:, 0] + bboxes_out[:, 2], \
        #              -bboxes_out[:, 1] + bboxes_out[:, 3]
        x, y, w, h = bboxes_out[:, 0], bboxes_out[:, 1], bboxes_out[:, 2], bboxes_out[:, 3]

        # make coordinates to be offset to default boxes and encode variance for xy and wh
        bboxes_out[:, 0] = (x - self.dboxes_xywh[:, 0]) / (self.scale_xy * self.dboxes_xywh[:, 2])
        bboxes_out[:, 1] = (y - self.dboxes_xywh[:, 1]) / (self.scale_xy * self.dboxes_xywh[:, 3])
        bboxes_out[:, 2] = torch.log(w / self.dboxes_xywh[:, 2]) / self.scale_wh
        bboxes_out[:, 3] = torch.log(h / self.dboxes_xywh[:, 3]) / self.scale_wh
        return bboxes_out, labels_out

    def decode(self, bboxes):
        xy = self.dboxes_xywh[:, :2]
        wh = self.dboxes_xywh[:, 2:]

        # don't forget that we need to remove variance from output
        _xy = (bboxes[:, :2] * self.scale_xy * wh) + xy
        _wh2 = (torch.exp(bboxes[:, 2:] * self.scale_wh) * wh) / 2
        xy1 = (_xy - _wh2)
        xy2 = (_xy + _wh2)
        boxes = torch.cat([xy1, xy2], dim=-1)

        return boxes


class OptimizedEncoder:
    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh")
        self.batch_dboxes = self.dboxes_xywh.cuda().unsqueeze(0)
        self.nboxes = self.dboxes.size(0)

        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, batch_size, bboxes, bboxes_offset, labels_in):
        bbox, label = C.box_encoder(batch_size, bboxes, bboxes_offset, labels_in, self.dboxes.cuda(), 0.5)
        # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        M = bbox.shape[0] // batch_size
        bboxes_out = bbox.view(batch_size, M, 4)
        labels_out = label.view(batch_size, M)

        x, y, w, h = bboxes_out[:, :, 0], bboxes_out[:, :, 1], bboxes_out[:, :, 2], bboxes_out[:, :, 3]

        bboxes_out[:, :, 0] = (x - self.batch_dboxes[:, :, 0]) / (self.scale_xy * self.batch_dboxes[:, :, 2])
        bboxes_out[:, :, 1] = (y - self.batch_dboxes[:, :, 1]) / (self.scale_xy * self.batch_dboxes[:, :, 3])
        bboxes_out[:, :, 2] = torch.log(w / self.batch_dboxes[:, :, 2]) / self.scale_wh
        bboxes_out[:, :, 3] = torch.log(h / self.batch_dboxes[:, :, 3]) / self.scale_wh

        return bboxes_out, labels_out
