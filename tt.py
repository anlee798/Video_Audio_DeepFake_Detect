import torch
import torch.nn as nn
import numpy as np
# from model2 import Audio_RNN
#
# model = Audio_RNN(img_dim=224, network='resnet34')
#
# video_input = torch.randn(4, 1, 3, 30, 224, 224)
# audio_input = torch.randn(4, 1, 1, 13, 99)
#
# final_out, vid_out_feat, aud_out_feat, vid_class, aud_class =model(video_input,audio_input)
# print(final_out.shape)
loss = nn.CrossEntropyLoss()
inputs = torch.randn(4,2)
arr = np.array([0,1,0,1])
# 直接转换为张量
targets = torch.tensor(arr) # torch.Size([4])
loss_v = loss(inputs,targets)
print("交叉熵：", loss_v)

inputs2 = torch.randn(4,1)     # input (4,1)
loss2 = nn.BCEWithLogitsLoss() # target(4,1)
arr = np.array([[0],
                [1],
                [0],
                [1]])
targets = torch.tensor(arr, dtype=torch.float32)
loss_v = loss2(inputs2, targets)
print("二元交叉熵：", loss_v)


# 模型的logits输出
inputs = torch.tensor([[0.7,0.2,0.1]]) # [1, 3]
# 真实标签                              #  [1]
arr = [1]
targets = torch.tensor(arr, dtype=torch.long)
# 创建交叉熵损失函数
loss = nn.CrossEntropyLoss()
# 计算损失
loss_v = loss(inputs, targets)
print("交叉熵损失：", loss_v.item())

