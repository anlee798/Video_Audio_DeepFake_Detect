import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFusionModel(nn.Module):
    def __init__(self):
        super(SimpleFusionModel, self).__init__()

        # 视频分支
        self.video_conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )
        self.video_fc = nn.Sequential(
            nn.Linear(32 * 128 * 24 * 24, 512),  # 假设输入视频尺寸 (3, 512, 96, 96)
            nn.ReLU(),
        )

        # 音频分支
        self.audio_fc = nn.Sequential(
            nn.Linear(64 * 2048, 512),
            nn.ReLU(),
        )

        # 融合分支
        self.fc_fusion = nn.Sequential(
            nn.Linear(512 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, videos, audios):
        # 视频分支
        video_features = self.video_conv(videos)
        video_features = video_features.view(video_features.size(0), -1)
        video_features = self.video_fc(video_features)

        # 音频分支
        audio_features = audios.view(audios.size(0), -1)
        audio_features = self.audio_fc(audio_features)

        # 融合特征
        fusion_features = torch.cat((video_features, audio_features), dim=1)
        output = self.fc_fusion(fusion_features)

        return output


# # 实例化模型
# device = 'cpu'
# model = SimpleFusionModel().to(device)
#
# # 示例输入
# videos = torch.randn(3, 3, 512, 96, 96).to(device)
# audios = torch.randn(3, 64, 2048).to(device)
# output = model(videos, audios)
#
# # 打印输出
# print("output", output)
# print("output shape", output.shape)
