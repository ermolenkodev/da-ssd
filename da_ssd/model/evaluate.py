import cv2
import torch
import time
import numpy as np
from contextlib import redirect_stdout
import io

from pycocotools.cocoeval import COCOeval


def denorm(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (x * std) + mean


def decode(dboxes_xywh, imgs, bboxes, labels):
    _xy = bboxes[:, :2]
    _wh2 = bboxes[:, 2:]
    xy1 = (_xy - 0.5*_wh2)
    xy2 = (_xy + 0.5*_wh2)
    boxes = torch.cat([xy1, xy2], dim=-1)

    for i in range(len(boxes)):
        img = imgs[i]
        box = boxes[i]
        label = labels[i]
        img = np.ascontiguousarray(denorm(img.cpu().numpy().transpose(1, 2, 0).copy()) * 255).astype(np.uint8)
        w, h = img.shape[:2]

        mask = (label > 0)
        res = box[mask].cpu().numpy()
        for bb in res:
            img = cv2.rectangle(img, (int(bb[0] * w), int(bb[1] * h)), (int(bb[2] * w), int(bb[3] * h)), (0, 255, 0), 2)

    cv2.imwrite(f'tmp_{i}.jpg', img)
        # plt.imshow(img)
        # plt.show()


def evaluate(model, coco, cocoGt, encoder, inv_map, args):
    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    model.eval()
    if not args.no_cuda:
        model.cuda()
    ret = []
    start = time.time()

    # for idx, image_id in enumerate(coco.img_keys):
    for nbatch, (img, img_id, img_size, _, _) in enumerate(coco):
        print("Parsing batch: {}/{}".format(nbatch, len(coco)), end='\r')
        with torch.no_grad():
            inp = img.cuda()
            if args.amp:
                inp = inp.half()

            # Get predictions
            ploc, plabel, _ = model(inp)
            ploc, plabel = ploc.float(), plabel.float()

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, 0.30, 200)[0]
                except:
                    # raise
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue

                # decode(_, inp, ploc_i, plabel_i)

                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id[idx], loc_[0] * wtot, \
                                loc_[1] * htot,
                                (loc_[2] - loc_[0]) * wtot,
                                (loc_[3] - loc_[1]) * htot,
                                prob_,
                                inv_map[label_]])

    # Now we have all predictions from this rank, gather them all together
    # if necessary
    ret = np.array(ret).astype(np.float32)

    # Multi-GPU eval
    if args.distributed:
        # NCCL backend means we can only operate on GPU tensors
        ret_copy = torch.tensor(ret).cuda()
        # Everyone exchanges the size of their results
        ret_sizes = [torch.tensor(0).cuda() for _ in range(N_gpu)]

        torch.cuda.synchronize()
        torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).cuda())
        torch.cuda.synchronize()

        # Get the maximum results size, as all tensors must be the same shape for
        # the all_gather call we need to make
        max_size = 0
        sizes = []
        for s in ret_sizes:
            max_size = max(max_size, s.item())
            sizes.append(s.item())

        # Need to pad my output to max_size in order to use in all_gather
        ret_pad = torch.cat([ret_copy, torch.zeros(max_size - ret_copy.shape[0], 7, dtype=torch.float32).cuda()])

        # allocate storage for results from all other processes
        other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).cuda() for i in range(N_gpu)]
        # Everyone exchanges (padded) results

        torch.cuda.synchronize()
        torch.distributed.all_gather(other_ret, ret_pad)
        torch.cuda.synchronize()

        # Now need to reconstruct the _actual_ results from the padded set using slices.
        cat_tensors = []
        for i in range(N_gpu):
            cat_tensors.append(other_ret[i][:sizes[i]][:])

        final_results = torch.cat(cat_tensors).cpu().numpy()
    else:
        # Otherwise full results are just our results
        final_results = ret

    if args.local_rank == 0:
        print("")
        print("Predicting Ended, total time: {:.2f} s".format(time.time() - start))

    cocoDt = cocoGt.loadRes(final_results)

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    if args.local_rank == 0:
        E.summarize()
        print("Current AP: {:.5f}".format(E.stats[0]))
    else:
        # fix for cocoeval indiscriminate prints
        with redirect_stdout(io.StringIO()):
            E.summarize()

    # put your model in training mode back on
    model.train()

    return E.stats[0]  # Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]
