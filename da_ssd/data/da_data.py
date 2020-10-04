import os

import torch
from torch.utils.data import DataLoader

from da_ssd.utils import dboxes300_coco, COCODetection
from da_ssd.utils import SSDTransformer
from da_ssd.data.coco import COCO
#DALI import
from da_ssd.data.coco_pipeline import COCOPipeline, DALICOCOIterator, SimplePipeline, DALIImageIterator


def get_train_loader(args, local_seed, size):
    # train_annotate = os.path.join(args.data, f"instances_train{args.coco_year}.json")
    # train_coco_root = os.path.join(args.data, f"train{args.coco_year}")

    train_annotate = args.train_annotations
    train_coco_root = args.train_data

    train_pipe = COCOPipeline(args.batch_size, args.local_rank, train_coco_root,
                    train_annotate, args.N_gpu, num_threads=args.num_workers,
                    output_fp16=args.amp, output_nhwc=False,
                    pad_output=False, seed=local_seed)
    train_pipe.build()
    test_run = train_pipe.schedule_run(), train_pipe.share_outputs(), train_pipe.release_outputs()
    train_loader = DALICOCOIterator(train_pipe, size / args.N_gpu)
    return train_loader


def get_val_dataset(args):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    # val_annotate = os.path.join(args.data, f"instances_val{args.coco_year}.json")
    # val_coco_root = os.path.join(args.data, f"val{args.coco_year}")

    val_annotate = args.test_annotations
    val_coco_root = args.test_data

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco


def get_val_dataloader(dataset, args):
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        val_sampler = None

    val_dataloader = DataLoader(dataset,
                                batch_size=args.eval_batch_size,
                                shuffle=False,  # Note: distributed sampler is shuffled :(
                                sampler=val_sampler,
                                num_workers=args.num_workers)

    return val_dataloader


def get_coco_ground_truth(args):
    val_annotate = args.test_annotations
    cocoGt = COCO(annotation_file=val_annotate)
    return cocoGt


def get_target_loader(args, local_seed, size):
    # root = os.path.join(args.data, "target")
    root = args.target_data
    # print(root)
    # train_pipe = SimplePipeline(root, args.batch_size // 2, num_threads=args.num_workers,
    #                             output_fp16=args.amp,
    #                             device_id=args.local_rank,
    #                             seed=local_seed)

    train_pipe = SimplePipeline(root, args.batch_size, num_threads=args.num_workers,
                                output_fp16=args.amp,
                                device_id=args.local_rank,
                                seed=local_seed)

    train_pipe.build()
    test_run = train_pipe.schedule_run(), train_pipe.share_outputs(), train_pipe.release_outputs()
    loader = DALIImageIterator(train_pipe, size / args.N_gpu)
    return loader
