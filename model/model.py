from typing import Dict, Optional, Union, Sequence, Tuple, Any

import torch
# from pytorch_lightning import LightningModule
# from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from dataset.lavdf import Metadata
from .audio_encoder import get_audio_encoder
from .boundary_module import BoundaryModule
from .frame_classifier import FrameLogisticRegression
from .fusion_module import ModalFeatureAttnBoundaryMapFusion
from .video_encoder import get_video_encoder
import torchmetrics
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.init as init

def sk_accuracy(output, target):
    with torch.no_grad():
        y_true_all, y_pred_all = np.array(target.cpu()), np.array(output.cpu())
        acc = accuracy_score(y_true_all, np.where(y_pred_all >= 0.5, 1, 0))*100.
        # acc = accuracy_score(y_true_all, target>=0.5)*100.
        return acc

class BCEFocalLoss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smoothing=0.01):
        super(BCEFocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.off_value = smoothing
        self.on_value = 1. - smoothing

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0]
        input = input.float().reshape(-1)
        target = target.float().reshape(-1)

        target = torch.where(target > 0.5, self.on_value, self.off_value)
        # print("target2", target.shape)
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class Batfd(nn.Module):

    def __init__(self,
        v_encoder: str = "c3d", a_encoder: str = "cnn", frame_classifier: str = "lr",
        ve_features=(64, 96, 128, 128), ae_features=(32, 64, 64), v_cla_feature_in=256, a_cla_feature_in=256,
        boundary_features=(512, 128), boundary_samples=10, temporal_dim=512, max_duration=40,
    ):
        super().__init__()
        self.cla_feature_in = v_cla_feature_in
        self.temporal_dim = temporal_dim

        self.video_encoder = get_video_encoder(v_cla_feature_in, temporal_dim, v_encoder, ve_features)
        self.audio_encoder = get_audio_encoder(a_cla_feature_in, temporal_dim, a_encoder, ae_features)

        if frame_classifier == "lr":
            self.video_frame_classifier = FrameLogisticRegression(n_features=v_cla_feature_in)
            self.audio_frame_classifier = FrameLogisticRegression(n_features=a_cla_feature_in)

        assert self.video_encoder and self.audio_encoder and self.video_frame_classifier and self.audio_frame_classifier

        assert v_cla_feature_in == a_cla_feature_in

        v_bm_in = v_cla_feature_in + 1
        a_bm_in = a_cla_feature_in + 1

        self.video_boundary_module = BoundaryModule(v_bm_in, boundary_features, boundary_samples, temporal_dim,
            max_duration
        )
        self.audio_boundary_module = BoundaryModule(a_bm_in, boundary_features, boundary_samples, temporal_dim,
            max_duration
        )

        self.fusion = ModalFeatureAttnBoundaryMapFusion(v_bm_in, a_bm_in, max_duration)

        self.embed = 512
        self.classifier = nn.Sequential(nn.Linear(self.embed, self.embed), nn.ReLU(inplace=True),
                                           nn.Linear(self.embed, 1))
        # self.mm_cls = BCEWithLogitsLoss()
        self.cls_loss = BCEFocalLoss2(alpha=0.55, gamma=2.5, smoothing=0.01)
        self.acc = torchmetrics.classification.BinaryAccuracy()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, video: Tensor, audio: Tensor) -> Tensor:
        # print("video.shape", video.shape) # video.shape torch.Size([16, 3, 512, 96, 96])
        # print("audio.shape", audio.shape) # audio.shape torch.Size([16, 64, 2048])
        # encoders
        v_features = self.video_encoder(video) #(1,256,512)
        a_features = self.audio_encoder(audio) #(1,256,512)

        # frame classifiers
        v_frame_cla = self.video_frame_classifier(v_features) #(1,1,512)
        a_frame_cla = self.audio_frame_classifier(a_features) #(1,1,512)

        # concat classification result to features
        v_bm_in = torch.column_stack([v_features, v_frame_cla]) #(1,257,512)
        a_bm_in = torch.column_stack([a_features, a_frame_cla]) #(1,257,512)

        # modal boundary module
        v_bm_map = self.video_boundary_module(v_bm_in) #(1,40,512)
        a_bm_map = self.audio_boundary_module(a_bm_in) #(1,40,512)

        # boundary map modal attention fusion
        fusion_bm_map = self.fusion(v_bm_in, a_bm_in, v_bm_map, a_bm_map) #(1,40,512)

        # print("fusion_bm_map", fusion_bm_map.shape)
        output = self.classifier(fusion_bm_map[:, 0, :]) #(1,1) # fusion_bm_map torch.Size([3, 40, 512])

        # return fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, v_features, a_features
        return output