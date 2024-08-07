# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""Train and eval functions used in main.py."""
import math
import sys
from typing import Iterable, Optional

import torch
import time
import datetime
import numpy as np
import random
from timm.utils import ModelEma, AverageMeter
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from utils import sk_accuracy, reduce_tensor
def calc_loss(vid_out, aud_out, target, hyper_param):
    batch_size = target.size(0)
    loss = 0
    for batch in range(batch_size):
        dist = torch.dist(vid_out[batch,:].view(-1), aud_out[batch,:].view(-1), 2)
        tar = target[batch,:].view(-1)
        loss += ((tar*(dist**2)) + ((1-tar)*(max(hyper_param-dist,0)**2)))
    return loss.mul_(1/batch_size)
def train_one_epoch(
        args,
        model,
        criterion,
        data_train_loader,
        optimizer,
        lr_scheduler,
        epoch,
        loss_scaler,
        model_ema,
        amp_autocast,
        device=torch.device('cuda'),
        logger=None,

):
    """train one epoch function."""
    model.train()
    criterion.train()
    optimizer.zero_grad()

    print_freq = 10
    num_steps = len(data_train_loader)
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backword_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    data_start_time = time.time()
    start = time.time()
    end = time.time()

    for batch_idx, (videos, audios, targets) in enumerate(data_train_loader):
        videos, audios, targets = videos.to(device), audios.to(device), targets.to(device)
        # print("videos.shape", videos.shape)
        # print("audios.shape", audios.shape)
        # print("targets.shape", targets.shape) # targets.shape torch.Size([4, 1])
        data_time.update(time.time() - data_start_time)

        torch.cuda.synchronize()
        start_forward = time.time()
        with amp_autocast():
            outputs,vid_out_feat, aud_out_feat, vid_class, aud_class = model(videos, audios)
            # print("outputs", outputs.shape) # outputs torch.Size([4, 2])
            # print("targets",targets.shape)  # targets.shape torch.Size([4, 1])
            # loss = criterion(outputs, targets)
            loss0 = criterion(outputs, targets.view(-1))
            loss1 = calc_loss(vid_out_feat, aud_out_feat, targets, 0.99)
            loss2 = criterion(vid_class, targets.view(-1))
            loss3 = criterion(aud_class, targets.view(-1))
            loss = loss0 * 3. + loss1 + loss2 + loss3
            # 计算准确率 应用 softmax 函数
            probabilities = torch.softmax(outputs, dim=1)
            # 选择概率最高的类别作为预测类别
            predicted_classes = probabilities.argmax(dim=1)
            # 将 targets 的形状从 [batch_size, 1] 转换为一维 [batch_size]
            true_classes = targets.squeeze(1)
            # 计算准确率
            correct_predictions = (predicted_classes == true_classes).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel()
            acc1 = accuracy.item()
            print("acc1", acc1)
        loss /= args.accumulation_steps
        loss_value = loss.item()
        torch.cuda.synchronize()
        end_forward = time.time()
        forward_time.update(end_forward - start_forward)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            optimizer.zero_grad()
            sys.exit(-1)

        if args.disable_amp:
            loss.backward()
            optimizer.step()
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(batch_idx + 1) % args.accumulation_steps == 0
            )
        torch.cuda.synchronize()
        backword_time.update(time.time() - end_forward)

        lr_scheduler.step_update(epoch * num_steps + batch_idx)
        optimizer.zero_grad()

        if model_ema is not None:
            model_ema.update(model)

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            loss_meter.update(reduced_loss.item(), targets.size(0))
        else:
            loss_meter.update(loss.item(), targets.size(0))
        # torch.distributed.barrier()

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - batch_idx)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{batch_idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.7f}\t wd {wd:.6f}\t'
                f'data-time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'forward-time {forward_time.val:.4f} ({forward_time.avg:.4f})\t'
                f'backward-time {backword_time.val:.4f} ({backword_time.avg:.4f})\t'
                f'batch-time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

        data_start_time = time.time()

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def evaluate(args, data_loader, model, criterion, amp_autocast, device, logger):
    """evaluation function."""

    # switch to evaluation mode
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    num_steps = len(data_loader)
    end = time.time()

    scores = []
    gt_labels = []
    with torch.no_grad():
        for batch_idx, (videos, audios, targets) in enumerate(data_loader):
            videos, audios, targets = videos.to(device), audios.to(device), targets.to(device)
            with amp_autocast():
                output, vid_out_feat, aud_out_feat, vid_class, aud_class = model(videos, audios)
                # loss = criterion(output.reshape(-1), targets)
                loss0 = criterion(output, targets.view(-1))
                loss1 = calc_loss(vid_out_feat, aud_out_feat, targets, 0.99)
                loss2 = criterion(vid_class, targets.view(-1)) # nn.CrossEntropyLoss()
                loss3 = criterion(aud_class, targets.view(-1)) # nn.CrossEntropyLoss()
                loss = loss0 * 3. + loss1 + loss2 + loss3

            # acc1 = torch.tensor(sk_accuracy(torch.sigmoid(output), targets), device=targets.device)
            # 应用 softmax 函数
            probabilities = torch.softmax(output, dim=1)
            # 选择概率最高的类别作为预测类别
            predicted_classes = probabilities.argmax(dim=1)
            # 将 targets 的形状从 [batch_size, 1] 转换为一维 [batch_size]
            true_classes = targets.squeeze(1)
            # 计算准确率
            correct_predictions = (predicted_classes == true_classes).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel() * 100
            # acc1 = (accuracy.item() * 100)
            acc1 = accuracy
            print("acc1", acc1)

            scores.extend(torch.sigmoid(output).reshape(-1).cpu().numpy().tolist())
            gt_labels.extend(targets.cpu().numpy().tolist())
            # acc1 = sk_accuracy(torch.sigmoid(output), target)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()
            loss_meter.update(reduced_loss.item(), targets.size(0))
            acc1_meter.update(acc1.item(), targets.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and batch_idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - batch_idx)
                logger.info(
                    f'Test: [{batch_idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))}\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

    scores = np.array(scores)
    gt_labels = np.array(gt_labels)
    thresh = 0.5
    logger.info(f'thresh: {thresh}')
    # logger.info(classification_report(gt_labels, scores > thresh, digits=3))

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}  * Val Loss  {loss_meter.avg:.4f}')
    return acc1_meter.avg, loss_meter.avg
