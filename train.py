# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------
# Modified from DeiT (https://github.com/facebookresearch/deit)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# --------------------------------------------------------------------------------

import os
import argparse
import datetime
import json
import time
from pathlib import Path
import torch.distributed as dist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, ModelEmaV2, get_state_dict, NativeScaler
from contextlib import suppress
from functools import partial

from torch import nn

import utils
from engine import evaluate, train_one_epoch
from logger import create_logger
# from data.dataset import build_dataloader
from mydeepfakedata.MultiDataset import build_dataloader
# from mpvit import DeepFakeClassifier_mpvit, load_pretrained_checkpoint2
from model2 import Audio_RNN
from model3 import Audio_RNN_ResNext
from lr_scheduler import build_scheduler
from loss import BCEFocalLoss, BCEFocalLoss2, BinaryCrossEntropy
# from lion_pytorch import Lion
import random

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


def get_args_parser():
    """
    get argugment parser.
    """
    parser = argparse.ArgumentParser("Video-Audio DeepFake Detect training and evaluation script", add_help=False)
    # Debug parameters
    parser.add_argument("--debug", action="store_true", help="enable debug mode")

    parser.add_argument('--train_data_path',
                        default='/data/zhuanlei/phase1',
                        type=str, help='path to dataset')
    parser.add_argument('--val_data_path',
                        default='/data/zhuanlei/phase1',
                        type=str, help='path to dataset')
    parser.add_argument('--augment', default=True, help='')
    # Basic training parameters.
    parser.add_argument("--batch-size", default=80, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--save_freq", default=1, type=int)

    parser.add_argument("--disable_amp", action="store_true", default=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="mpvit_base",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input-size", default=224, type=int, help="images input size")
    parser.add_argument("--num-classes", default=1, type=int, help="")

    parser.add_argument("--prefetcher", type=bool, default=True, help="fast prefetcher")

    parser.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                        help="Enable compilation w/ specified backend (default: inductor).")

    parser.add_argument("--model-ema", default=False, action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument("--model-ema-force-cpu", action="store_true", default=False, help="")

    # Optimizer parameters
    # parser.add_argument("--lr", type=float, default=1e-5, metavar="LR", help="learning rate (default: 5e-4)")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR", help="learning rate (default: 5e-4)") # default=1e-5
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw"')
    parser.add_argument("--opt-eps", default=1e-8, type=float, metavar="EPSILON",
                        help="Optimizer Epsilon (default: 1e-8)")
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+", metavar="BETA",
                        help="Optimizer Betas (default: None, use opt default)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
    parser.add_argument("--weights-decay", type=float, default=1e-5, help="weights decay (default: 0.05)")

    # Learning rate schedule parameters
    parser.add_argument("--sched", default="cosine", type=str, metavar="SCHEDULER", # default="cosine"
                        help='LR scheduler (default: "cosine"')
    parser.add_argument("--lr-noise", type=float, nargs="+", default=None, metavar="pct, pct", # default=None
                        help="learning rate noise on/off epoch percentages")
    parser.add_argument("--lr-noise-pct", type=float, default=0.67, metavar="PERCENT", # default=0.67
                        help="learning rate noise limit percent (default: 0.67)")
    parser.add_argument("--lr-noise-std", type=float, default=1.0, metavar="STDDEV", # default=1.0
                        help="learning rate noise std-dev (default: 1.0)")
    parser.add_argument("--warmup-lr", type=float, default=1e-3, metavar="LR", # default=1e-7
                        help="warmup learning rate (default: 1e-6)")
    parser.add_argument("--min-lr", type=float, default=1e-6, metavar="LR", # default=1e-7
                        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)")

    parser.add_argument("--decay-epochs", type=float, default=30, metavar="N", help="epoch interval to decay LR") # default=30
    parser.add_argument("--warmup-epochs", type=int, default=0, metavar="N", # default=0
                        help="epochs to warmup LR, if scheduler supports")
    parser.add_argument("--cooldown-epochs", type=int, default=10, metavar="N",
                        help="epochs to cooldown LR at min_lr, after cyclic schedule ends")
    parser.add_argument("--patience-epochs", type=int, default=10, metavar="N",
                        help="patience epochs for Plateau LR scheduler (default: 10")
    parser.add_argument("--decay-rate", "--dr", type=float, default=0.1, metavar="RATE",
                        help="LR decay rate (default: 0.1)")

    parser.add_argument("--smoothing", type=float, default=0.01, help="Label smoothing (default: 0.1)")

    parser.add_argument('--clip-grad', type=float, default=1., metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')

    parser.add_argument('--accumulation-steps', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', default=True, action='store_true',
                        help="whether to use gradient checkpointing to save memory")

    parser.add_argument("--output_dir",
                        default="./weights/dataset_dh/audio_rnn/audio_rnn_resnet34_lr_1e3",
                        help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume",
                        default="/data0/gj/project/ps_project/deepfake/project_classfier/MPViT-main/weights/dataset_dh/v2.2/mpvit_base_384_fl-a0.5-g2.5_160-128/checkpoint_epoch_5_99.3362508355615.pth",
                        # default="",
                        help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=4, type=int)  # Note: Original 10 is very high.

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')

    parser.add_argument("--gpu", nargs='+', type=int, default=[0, 1, 2], help="List of GPU ids")
    return parser


def main(args):
    """
    training main function.
    """

    utils.init_distributed_mode(args)

    # logger = create_logger(output_dir=args.output_dir, dist_rank=dist.get_rank())
    logger = create_logger(output_dir=args.output_dir, dist_rank=args.gpu)
    logger.info(args)

    # Debug mode.
    if args.debug:
        import debugpy
        logger.info("Enabling attach starts.")
        debugpy.listen(address=("0.0.0.0", 9310))
        debugpy.wait_for_client()
        logger.info("Enabling attach ends.")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # logger.info(f"Creating model: {args.model}")
    logger.info(f"Creating model: Aduio_RNN")
    # Model
    # model = DeepFakeClassifier_mpvit(args.model, num_classes=args.num_classes, grad_checkpointing=args.use_checkpoint, pretrained=False,
    #                                  pretrained_path='./weights/mpvit_base.pth')
    model = Audio_RNN(img_dim=224, network='resnet34')
    model.to(device)
    # model = Audio_RNN_ResNext(img_dim=224)
    # model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters/1e6}")

    has_compile = hasattr(torch, 'compile')
    if args.torchcompile and has_compile:
        # torch compile should be done after DDP
        #### mode: reduce-overhead, max-autotune  编译时需要更大的显存
        model = torch.compile(model, backend=args.torchcompile)
        # model = torch.compile(model, mode="reduce-overhead", backend=args.torchcompile)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr
    args.lr = args.lr
    args.weight_decay = 1e-5
    optimizer = create_optimizer(args, model)
    # optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if not args.disable_amp:
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)
        loss_scaler = utils.NativeScalerWithGradAccumulation()

    train_data_loader, val_data_loader = build_dataloader(args, device=device, logger=logger)

    # n_iter_per_epoch = min([len(dataloader) for dataloader in [train_pos_data_loader, train_neg_data_loader]])
    n_iter_per_epoch = len(train_data_loader)
    print("n_iter_per_epoch", n_iter_per_epoch)
    lr_scheduler = build_scheduler(args, optimizer, n_iter_per_epoch)

    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = BCEFocalLoss(alpha=0.5, gamma=2.5, smoothing=args.smoothing)
    # criterion = BCEFocalLoss2(alpha=0.55, gamma=2.5, smoothing=args.smoothing) # MpViT
    # criterion = BinaryCrossEntropy(smoothing=args.smoothing)
    # criterion = BinaryCrossEntropy(smoothing=args.smoothing, pos_weight=torch.tensor([1.5],device=device))
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss

    output_dir = args.output_dir

    # if args.resume:
    #     logger.info(f"resume from {args.resume}")
    #     if args.resume.startswith("https"):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.resume, map_location="cpu", check_hash=True
    #         )
    #     else:
    #         checkpoint = torch.load(args.resume, map_location="cpu")
    #     model_without_ddp.load_state_dict(checkpoint["model"])
    #     if (
    #             not args.eval
    #             and "optimizer" in checkpoint
    #             and "lr_scheduler" in checkpoint
    #             and "epoch" in checkpoint
    #     ):
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    #         lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    #         args.start_epoch = checkpoint["epoch"] + 1
    #         # if args.model_ema:
    #         #     utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])

    logger.info("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_data_loader.sampler.set_epoch(epoch)

        train_one_epoch(
            args,
            model,
            criterion,
            train_data_loader,
            optimizer,
            lr_scheduler,
            epoch,
            loss_scaler,
            model_ema,
            amp_autocast,
            device=device,
            logger=logger
        )

        if epoch % args.save_freq == 0:
            acc1, _ = evaluate(args, val_data_loader, model, criterion, amp_autocast, device, logger=logger)
            logger.info(
                f"Accuracy of the network on the {len(val_data_loader)} test images: {acc1:.1f}%"
            )
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f"Max accuracy: {max_accuracy:.2f}%")

            checkpoint_path = os.path.join(output_dir, f'Audio_RNN_checkpoint_epoch_{epoch}_{acc1}.pth')
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    # "model_ema": get_state_dict(model_ema),
                    "args": args,
                },
                checkpoint_path,
            )

            save_path = os.path.join(output_dir, f'Audio_RNN_ckpt_epoch_{epoch}_{acc1}_{args.gpu}.pth.tar')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'epoch': epoch,
                'acc1': acc1
            }, save_path)

        torch.cuda.empty_cache()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Multi_DeepFake_Video_Audio training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# torchrun --standalone --nproc_per_node=3 train.py --gpu 0 1 2
