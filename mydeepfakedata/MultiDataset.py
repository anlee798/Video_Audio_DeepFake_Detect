from torch.utils.data import Dataset
from .utils_video_to_audio_video import load_video_audio_to_tensor
import os
import pandas as pd
import warnings
import torch
import random
import numpy as np

warnings.filterwarnings("ignore")


class MultiDeepFakeDatasets(Dataset):
    def __init__(self,
                 args,
                 mode='train',
                 device='cpu'):
        super().__init__()
        self.decive = device
        video_info = None
        if mode == 'train':
            split = os.path.join(args.train_data_path, 'train_list.csv')
            # split = os.path.join('./mydeepfakedata', 'train_list.csv') #本地小批量数据
            video_info = pd.read_csv(split, header=1)
        elif mode == 'val':
            split = os.path.join(args.val_data_path, 'val_list.csv')
            # split = os.path.join('./mydeepfakedata', 'val_list.csv') #本地小批量数据
            video_info = pd.read_csv(split, header=1)
        self.video_info = video_info

    def __getitem__(self, index):
        vpath, audiopath, label = self.video_info.iloc[index]
        vpath_list = []
        vpath_list.append(vpath)
        audiopath_list = []
        audiopath_list.append(audiopath)
        video_tensor, audio_tensor = load_video_audio_to_tensor(video_paths=[vpath],
                                                                audio_paths=[audiopath],
                                                                device=self.decive,
                                                                num_mel_bins=13,
                                                                target_length=99,
                                                                target_video_fps=30,
                                                                # target_video_fps=16,
                                                                get_target_video_method='interpolate')
        # print(video_tensor.shape) # torch.Size([1, 1, 3, 30, 224, 224]) #ACM torch.Size([1, 1, 3, 30, 224, 224]) torch.Size([4, 1, 3, 30, 224, 224])
        # print(audio_tensor.shape) # torch.Size([1, 1, 1, 128, 204])     # #  torch.Size([1, 1, 1, 13, 99])       torch.Size([4, 1, 1, 13, 99])
        return video_tensor.squeeze(0), audio_tensor.squeeze(0), torch.as_tensor(label)

    def __len__(self):
        return len(self.video_info)

    def reset_seed(self, epoch, seed):
        seed = (epoch + 1) * seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def collate_fn(batch):
        video, audio, labels = tuple(zip(*batch))
        video = torch.stack(video, dim=0)
        audio = torch.stack(audio, dim=0)
        labels = torch.as_tensor(labels).unsqueeze(-1)
        # print(type(video), type(audio), type(labels))
        # # videos.shape torch.Size([4, 1, 1, 3, 30, 224, 224])
        # # audios.shape torch.Size([4, 1, 1, 1, 13, 99])
        # # targets.shape torch.Size([4])
        return video.squeeze(0), audio.squeeze(0), labels


def build_dataloader(args, device=torch.device('cuda'), logger=None):
    data_train = MultiDeepFakeDatasets(args, mode='train', device=device)
    data_train_sampler = None
    if args.distributed:
        data_train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)

    data_val = MultiDeepFakeDatasets(args, mode='val', device=device)
    data_val_sampler = None
    if args.distributed:
        data_val_sampler = torch.utils.data.distributed.DistributedSampler(data_val)

    train_data_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=data_train_sampler is None,
        pin_memory=True,
        sampler=data_train_sampler,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=data_train.collate_fn,
    )
    val_data_loader = torch.utils.data.DataLoader(
        data_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=data_val_sampler is None,
        pin_memory=True,
        sampler=data_val_sampler,
        drop_last=True,
        persistent_workers=False,
        collate_fn=data_val.collate_fn,
    )

    return train_data_loader, val_data_loader
