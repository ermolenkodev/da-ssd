import cv2
from torch.autograd import Variable
import torch
import time
from SSD import _C as C

from apex import amp

from da_ssd.model.bbox_encoding import BoxUtils
import numpy as np
from matplotlib import pyplot as plt


def denorm(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (x * std) + mean


def decode2(imgs, boxes):
    # _xy = bboxes[:, :, :2]
    # _wh2 = bboxes[:, :, 2:]
    # xy1 = (_xy - 0.5 * _wh2)
    # xy2 = (_xy + 0.5 * _wh2)
    # boxes = torch.cat([xy1, xy2], dim=-1)

    for i in range(len(boxes)):
        img = imgs[i]
        bb = boxes[i]
        # label = labels[i]
        img = np.ascontiguousarray(denorm(img.cpu().numpy().transpose(1, 2, 0).copy()) * 255).astype(np.uint8)
        w, h = img.shape[:2]

        # mask = (label > 0)
        # res = box[mask].cpu().numpy()
        img = cv2.rectangle(img, (int(bb[0] * w), int(bb[1] * h)), (int(bb[2] * w), int(bb[3] * h)),
                            (0, 255, 0), 2)

        cv2.imwrite(f'tmp_{i}.jpg', img)


def decode(dboxes_xywh, imgs, bboxes, labels):
    _xy = bboxes[:, :, :2]
    _wh2 = bboxes[:, :, 2:]
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


OTHERS = 2


def train_loop(model, loss_func, da_loss, epoch, optim, train_dataloader, target_dataloader,
               encoder, iteration, logger, args, mean, std, meters, vis):

    #     for nbatch, (img, _, img_size, bbox, label) in enumerate(train_dataloader):
    for nbatch, (source_data, target_data) in enumerate(zip(train_dataloader, target_dataloader)):
        target_img = target_data[0][0]
        img = source_data[0][0][0]
        bbox = source_data[0][1][0]
        label = source_data[0][2][0]
        # decode2(img, bbox)
        label = label.type(torch.cuda.LongTensor)
        batch_size = img.shape[0]
        source_domain_labels = torch.zeros(batch_size)
        # target_domain_labels = torch.ones(batch_size // 2)
        target_domain_labels = torch.ones(batch_size)
        bbox_offsets = source_data[0][3][0]
        # handle random flipping outside of DALI for now
        bbox_offsets = bbox_offsets.cuda()
        img, bbox = C.random_horiz_flip(img, bbox, bbox_offsets, 0.5, False)
        img.sub_(mean).div_(std)
        if not args.no_cuda:
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            source_domain_labels = source_domain_labels.cuda()
            target_domain_labels = target_domain_labels.cuda()
            target_img = target_img.cuda()
            # bbox_offsets = bbox_offsets.cuda()

        domain_label = torch.cat([source_domain_labels, target_domain_labels], dim=0)
        images = torch.cat([img, target_img], dim=0)

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue
        bbox, label = C.box_encoder(N, bbox, bbox_offsets, label, encoder.dboxes.cuda(), 0.5)
        # label = label * (label != OTHERS).long()
        # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)

        # bbox = torch.cat([bbox, bbox.new_ones((N // 2,) + bbox.shape[1:])], dim=0)
        # label = torch.cat([label, label.new_ones((N // 2,) + label.shape[1:])], dim=0)

        bbox = torch.cat([bbox, bbox.new_ones((N,) + bbox.shape[1:])], dim=0)
        label = torch.cat([label, label.new_ones((N,) + label.shape[1:])], dim=0)

        ploc, plabel, domain_classifier_features = model(images)
        ploc, plabel = ploc.float(), plabel.float()

        # decode(encoder.dboxes_xywh, images, bbox, label)

        trans_bbox = bbox.transpose(1, 2).contiguous().cuda()

        if not args.no_cuda:
            label = label.cuda()
        gloc = Variable(trans_bbox, requires_grad=False)
        glabel = Variable(label, requires_grad=False)

        ssd_loss = loss_func(ploc, plabel, gloc, glabel, domain_label)
        adaptation_loss = da_loss(domain_classifier_features, domain_label)

        # loss = ssd_loss + 10 * adaptation_loss
        loss = ssd_loss + adaptation_loss

        if args.amp:
            with amp.scale_loss(loss, optim) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()

        if args.warmup is not None:
            warmup(optim, args.warmup, iteration, args.learning_rate)

        optim.step()
        optim.zero_grad()

        if args.local_rank == 0:
            logger.update_iter(epoch, iteration, loss.item())
            meters['total'].add(loss.cpu().detach().numpy())
            meters['ssd'].add(ssd_loss.cpu().detach().numpy())
            meters['da'].add(adaptation_loss.cpu().detach().numpy())

            if (nbatch + 1) % args.plot_every == 0:
                # plot loss
                vis.plot_many({k: v.value()[0] for k, v in meters.items()})

        iteration += 1

    return iteration

def benchmark_train_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    start_time = None
    # tensor for results
    result = torch.zeros((1,)).cuda()
    for i, data in enumerate(loop(train_dataloader)):
        if i >= args.benchmark_warmup:
            start_time = time.time()

        img = data[0][0][0]
        bbox = data[0][1][0]
        label = data[0][2][0]
        label = label.type(torch.cuda.LongTensor)
        bbox_offsets = data[0][3][0]
        # handle random flipping outside of DALI for now
        bbox_offsets = bbox_offsets.cuda()
        img, bbox = C.random_horiz_flip(img, bbox, bbox_offsets, 0.5, False)

        if not args.no_cuda:
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            bbox_offsets = bbox_offsets.cuda()
        img.sub_(mean).div_(std)

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue
        bbox, label = C.box_encoder(N, bbox, bbox_offsets, label, encoder.dboxes.cuda(), 0.5)

        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)





        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()

        trans_bbox = bbox.transpose(1, 2).contiguous().cuda()

        if not args.no_cuda:
            label = label.cuda()
        gloc = Variable(trans_bbox, requires_grad=False)
        glabel = Variable(label, requires_grad=False)

        loss = loss_func(ploc, plabel, gloc, glabel)



        # loss scaling
        if args.amp:
            with amp.scale_loss(loss, optim) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()

        optim.step()
        optim.zero_grad()

        if i >= args.benchmark_warmup + args.benchmark_iterations:
            break

        if i >= args.benchmark_warmup:
            logger.update(args.batch_size, time.time() - start_time)


    result.data[0] = logger.print_result()
    if args.N_gpu > 1:
        torch.distributed.reduce(result, 0)
    if args.local_rank == 0:
        print('Training performance = {} FPS'.format(float(result.data[0])))



def loop(dataloader):
    while True:
        for data in dataloader:
            yield data

def benchmark_inference_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    assert args.N_gpu == 1, 'Inference benchmark only on 1 gpu'
    start_time = None
    model.eval()

    i = -1
    val_datas = loop(val_dataloader)

    while True:
        i += 1
        torch.cuda.synchronize()
        if i >= args.benchmark_warmup:
            start_time = time.time()

        data = next(val_datas)

        with torch.no_grad():
            img = data[0]
            if not args.no_cuda:
                img = img.cuda()
            if args.amp:
                img = img.half()
            img.sub_(mean).div_(std)
            img = Variable(img, requires_grad=False)
            _ = model(img)
            torch.cuda.synchronize()

            if i >= args.benchmark_warmup + args.benchmark_iterations:
                break

            if i >= args.benchmark_warmup:
                logger.update(args.eval_batch_size, time.time() - start_time)

    logger.print_result()

def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]
