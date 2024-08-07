# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import io
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from timm.utils.clip_grad import dispatch_clip_grad


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        """
        init function
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        update fuction
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """
        median fuction
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """
        avg function
        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """
        global average function
        """
        return self.total / self.count

    @property
    def max(self):
        """
        global average function
        """
        return max(self.deque)

    @property
    def value(self):
        """
        global average function
        """
        return self.deque[-1]

    def __str__(self):
        """
        to string
        """
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    """MetricLogger class"""

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        update fuction
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        sync
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        add_meter
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        logger
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_for_distributed2(rank):
    """
    This function disables printing when not in master process2
    """    
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        builtin_print("[RANK:{}]".format(rank), *args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    init
    """    
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    get world size
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    get rank
    """    
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    helper function
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    helper function
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    helper function
    """    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.gpu = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=5400)
    )
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)
    setup_for_distributed2(args.rank)

class NativeScalerWithGradAccumulation:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

import numpy as np
from sklearn.metrics import accuracy_score
def sk_accuracy(output, target):
    with torch.no_grad():
        y_true_all, y_pred_all = np.array(target.cpu()), np.array(output.cpu())
        acc = accuracy_score(y_true_all, np.where(y_pred_all >= 0.5, 1, 0))*100.
        # acc = accuracy_score(y_true_all, target>=0.5)*100.
        return acc

from importlib import metadata
import json
import os
import re
from abc import ABC
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
import torchvision
from einops import rearrange
# from pytorch_lightning import Callback, Trainer, LightningModule
from torch import Tensor
from torch.nn import functional as F, Module


def read_json(path: str, object_hook=None):
    with open(path, 'r') as f:
        return json.load(f, object_hook=object_hook)


def read_video(path: str):
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    print("audio.shape", audio.shape)
    video = video.permute(0, 3, 1, 2) / 255
    audio = audio.permute(1, 0)
    return video, audio, info


def read_audio(path: str):
    return torchaudio.load(path)


def read_image(path: str):
    return torchvision.io.read_image(path).float() / 255.0


def padding_video(tensor: Tensor, target: int, padding_method: str = "zero", padding_position: str = "tail") -> Tensor:
    t, c, h, w = tensor.shape
    padding_size = target - t

    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        return F.pad(tensor, pad=[0, 0, 0, 0, 0, 0] + pad)
    elif padding_method == "same":
        tensor = rearrange(tensor, "t c h w -> c h w t")
        tensor = F.pad(tensor, pad=pad + [0, 0], mode="replicate")
        return rearrange(tensor, "c h w t -> t c h w")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")


def padding_audio(tensor: Tensor, target: int,
    padding_method: str = "zero",
    padding_position: str = "tail"
) -> Tensor:
    t, c = tensor.shape
    padding_size = target - t
    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        return F.pad(tensor, pad=[0, 0] + pad)
    elif padding_method == "same":
        tensor = rearrange(tensor, "t c -> 1 c t")
        tensor = F.pad(tensor, pad=pad, mode="replicate")
        return rearrange(tensor, "1 c t -> t c")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")


def _get_padding_pair(padding_size: int, padding_position: str) -> List[int]:
    if padding_position == "tail":
        pad = [0, padding_size]
    elif padding_position == "head":
        pad = [padding_size, 0]
    elif padding_position == "average":
        padding_head = padding_size // 2
        padding_tail = padding_size - padding_head
        pad = [padding_head, padding_tail]
    else:
        raise ValueError("Wrong padding position. It should be zero or tail or average.")
    return pad


def resize_video(tensor: Tensor, size: Tuple[int, int], resize_method: str = "bicubic") -> Tensor:
    return F.interpolate(tensor, size=size, mode=resize_method)


class _ConvNd(Module, ABC):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
        build_activation: Optional[callable] = None
    ):
        super().__init__()
        self.conv = self.PtConv(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        if build_activation is not None:
            self.activation = build_activation()
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv1d(_ConvNd):
    PtConv = torch.nn.Conv1d


class Conv2d(_ConvNd):
    PtConv = torch.nn.Conv2d


class Conv3d(_ConvNd):
    PtConv = torch.nn.Conv3d


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors."""

    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    iou = inter_len / union_len
    return iou


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_1d(proposal, target) -> Tensor:
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The predicted array with [M, 2]. First column is
            beginning, second column is end.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The label array with [N, 2]. First column is
            beginning, second column is end.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is np.ndarray:
        proposal = torch.from_numpy(proposal)

    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.)
    union = outer_end - outer_begin
    return inter / union


# class LrLogger(Callback):
#     """Log learning rate in each epoch start."""
#
#     def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         for i, optimizer in enumerate(trainer.optimizers):
#             for j, params in enumerate(optimizer.param_groups):
#                 key = f"opt{i}_lr{j}"
#                 value = params["lr"]
#                 pl_module.logger.log_metrics({key: value}, step=trainer.global_step)
#                 pl_module.log(key, value, logger=False, sync_dist=pl_module.distributed)


# class EarlyStoppingLR(Callback):
#     """Early stop model training when the LR is lower than threshold."""
#
#     def __init__(self, lr_threshold: float, mode="all"):
#         self.lr_threshold = lr_threshold
#
#         if mode in ("any", "all"):
#             self.mode = mode
#         else:
#             raise ValueError(f"mode must be one of ('any', 'all')")
#
#     def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
#         self._run_early_stop_checking(trainer)
#
#     def _run_early_stop_checking(self, trainer: Trainer) -> None:
#         metrics = trainer._logger_connector.callback_metrics
#         if len(metrics) == 0:
#             return
#         all_lr = []
#         for key, value in metrics.items():
#             if re.match(r"opt\d+_lr\d+", key):
#                 all_lr.append(value)
#
#         if len(all_lr) == 0:
#             return
#
#         if self.mode == "all":
#             if all(lr <= self.lr_threshold for lr in all_lr):
#                 trainer.should_stop = True
#         elif self.mode == "any":
#             if any(lr <= self.lr_threshold for lr in all_lr):
#                 trainer.should_stop = True


def generate_metadata_min(data_root: str):
    metadata_full = read_json(os.path.join(data_root, "metadata.json"))
    metadata_min = []
    for meta in metadata_full:
        del meta["timestamps"]
        del meta["transcript"]
        metadata_min.append(meta)
    with open(os.path.join(data_root, "metadata.min.json"), "w") as f:
        json.dump(metadata_min, f)