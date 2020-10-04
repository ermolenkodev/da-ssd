import os
import time
from argparse import ArgumentParser
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data.distributed

from da_ssd.model.model import SSD300, ResNet, Loss, DASSD300, DALoss, ImageLevelAdaptationLoss
from da_ssd.utils import dboxes300_coco, Encoder
from da_ssd.model.logger import Logger, BenchLogger
from da_ssd.model.evaluate import evaluate
from da_ssd.model.train import train_loop, tencent_trick, load_checkpoint, benchmark_train_loop, benchmark_inference_loop
from da_ssd.data.da_data import get_train_loader, get_val_dataset, get_val_dataloader, get_coco_ground_truth, get_target_loader

# Apex imports
from da_ssd.visualisation import AverageValueMeter, Visualizer

try:
    from apex.parallel.LARC import LARC
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")


def generate_mean_std(args):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean_val).cuda()
    std = torch.tensor(std_val).cuda()

    view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    if args.amp:
        mean = mean.half()
        std = std.half()

    return mean, std


def make_parser():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    # parser.add_argument('--data', '-d', type=str, default='/data/coco/', required=False,
    #                     help='path to test and training data files')

    parser.add_argument('--train-data', '-train', type=str, default='/data/adaptation_dataset/synthetic', required=False,
                        help='path to training data files')
    parser.add_argument('--test-data', '-test', type=str, default='/data/crabs/val', required=False,
                        help='path to test data files')
    parser.add_argument('--target-data', '-target', type=str, default='/data/crabs/images/positive', required=False,
                        help='path to target dataset')

    parser.add_argument('--train-annotations', type=str, default='/data/adaptation_dataset/synthetic/train.json', required=False,
                        help='path to training data annotations')
    parser.add_argument('--test-annotations', type=str, default='/data/crabs/val/ann/train.json', required=False,
                        help='path to test data annotations')

    # parser.add_argument('--coco-year', '-y', type=str, default='2017', required=False,
    #                     help='COCO dataset year')

    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '--bs', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--eval-batch-size', '--ebs', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--save', action='store_true',
                        help='save model checkpoints')
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'evaluation', 'benchmark-training', 'benchmark-inference'])
    # parser.add_argument('--evaluation', nargs='*', type=int, default=[21, 31, 37, 42, 48, 53, 59, 64],
    #                     help='epochs at which to evaluate')
    parser.add_argument('--evaluation', nargs='*', type=int, default=[3, 6, 9, 15, 20, 25, 30, 35, 40, 45, 50],
                        help='epochs at which to evaluate')
    parser.add_argument('--multistep', nargs='*', type=int, default=[6, 15, 25, 35, 45],
                        help='epochs at which to decay learning rate')
    # parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54],
    #                     help='epochs at which to decay learning rate')

    # Hyperparameters
    parser.add_argument('--learning-rate', '--lr', type=float, default=2.6e-3,
                        help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum argument for SGD optimizer')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0.0005,
                        help='momentum argument for SGD optimizer')

    parser.add_argument('--profile', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--benchmark-iterations', type=int, default=20, metavar='N',
                        help='Run N iterations while benchmarking (ignored when training and validation)')
    parser.add_argument('--benchmark-warmup', type=int, default=20, metavar='N',
                        help='Number of warmup iterations for benchmarking')

    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--backbone-path', type=str, default=None,
                        help='Path to chekcpointed backbone. It should match the'
                             ' backbone model declared with the --backbone argument.'
                             ' When it is not provided, pretrained model from torchvision'
                             ' will be downloaded.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--plot-every', default=100)

    # Distributed
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')

    return parser


def train(train_loop_func, logger, args):
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda

    # Setup multi-GPU if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.N_gpu = torch.distributed.get_world_size()
    else:
        args.N_gpu = 1

    if args.seed is None:
        args.seed = np.random.randint(1e4)

    if args.distributed:
        args.seed = (args.seed + torch.distributed.get_rank()) % 2**32
    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    torch.multiprocessing.set_sharing_strategy('file_system')

    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)
    #82783
    # train_loader = get_train_loader(args, args.seed - 2**31, 118287)

    # target_loader = get_target_loader(args, args.seed - 2**31, 118287)

    train_loader = get_train_loader(args, args.seed - 2**31, 5000)

    target_loader = get_target_loader(args, args.seed - 2**31, 5000)

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    ssd300 = DASSD300(backbone=ResNet(args.backbone, args.backbone_path))
    # ?????args.learning_rate = args.learning_rate * args.N_gpu * ((args.batch_size + args.batch_size // 2) / 32)
    args.learning_rate = args.learning_rate * args.N_gpu * ((args.batch_size + args.batch_size) / 32)
    start_epoch = 0
    iteration = 0
    loss_func = DALoss(dboxes)
    da_loss_func = ImageLevelAdaptationLoss()

    if use_cuda:
        ssd300.cuda()
        loss_func.cuda()
        da_loss_func.cuda()

    optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
    if args.amp:
        ssd300, optimizer = amp.initialize(ssd300, optimizer, opt_level='O2')

    if args.distributed:
        ssd300 = DDP(ssd300)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300.module if args.distributed else ssd300, args.checkpoint)
            checkpoint = torch.load(args.checkpoint,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    inv_map = {v: k for k, v in val_dataset.label_map.items()}

    total_time = 0

    if args.mode == 'evaluation':
        acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
        if args.local_rank == 0:
            print('Model precision {} mAP'.format(acc))

        return
    mean, std = generate_mean_std(args)

    meters = {
        'total': AverageValueMeter(),
        'ssd': AverageValueMeter(),
        'da': AverageValueMeter()
    }

    vis = Visualizer(env='da ssd', port=6006)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        scheduler.step()
        iteration = train_loop_func(ssd300, loss_func, da_loss_func, epoch, optimizer, train_loader, target_loader, encoder, iteration,
                                    logger, args, mean, std, meters, vis)
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            logger.update_epoch_time(epoch, end_epoch_time)

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)

            if args.local_rank == 0:
                logger.update_epoch(epoch, acc)
                vis.log(acc, win='Evaluation')

        if args.save and args.local_rank == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'label_map': val_dataset.label_info}
            if args.distributed:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            torch.save(obj, './models/epoch_{}.pt'.format(epoch))
        train_loader.reset()
        target_loader.reset()

    print('total training time: {}'.format(total_time))


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.local_rank == 0:
        os.makedirs('./models', exist_ok=True)

    torch.backends.cudnn.benchmark = True

    train_loop_func = train_loop
    logger = Logger('Training logger', print_freq=1)

    train(train_loop_func, logger, args)
