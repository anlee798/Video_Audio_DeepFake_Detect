# -*- coding: utf-8 -*-
# author: AnLee

import torch
import torch.nn as nn

# 假设模型的输出，形状为 [batch_size, num_classes]
outputs = torch.randn(4, 2)  # 这里用随机数模拟模型输出

# 真实标签，形状为 [batch_size, 1]
targets = torch.randint(0, 2, (4, 1))  # 假设每个样本都是随机的0或1

# 将targets的形状从 [batch_size, 1] 转换为 [batch_size]
targets = targets.squeeze(1)
print("outputs",outputs)
print("targets",targets)
# 计算交叉熵损失
loss_ce = nn.CrossEntropyLoss()
ce_loss = loss_ce(outputs, targets)
print("ce_loss", ce_loss) # ce_loss tensor(0.8918)

# 计算二元交叉熵损失
# 首先，我们需要将模型的输出调整为适合二元交叉熵的形式
# 这里我们假设outputs的第二列是正类的得分，第一列是负类的得分
outputs_for_bce = outputs[:, 1].view(4, 1)  # 选择正类的得分

# 然后，我们需要将targets的形状从 [batch_size] 转换为 [batch_size, 1]
targets_for_bce = targets.unsqueeze(1)

# 现在，我们可以使用BCEWithLogitsLoss
loss_bce = nn.BCEWithLogitsLoss()
bce_loss = loss_bce(outputs_for_bce, targets_for_bce)

print("交叉熵损失:", ce_loss.item())
print("二元交叉熵损失:", bce_loss.item())

'''
如果你的模型输出 outputs 的形状是 torch.Size([4, 2])，这表明你有一个包含4个样本的批次，每个样本有两个类别的得分。而你的 targets 形状是 torch.Size([4, 1])，且包含的数值都是0或1，这意味着每个样本属于两个类别中的一个。

在这种情况下，如果你想计算二元交叉熵损失，你需要调整模型输出，使其与二元分类问题相匹配。由于 targets 中的数值是0或1，我们可以假设第一个类别的得分对应于 outputs 中的第二列（索引为1），而 targets 中的0将对应于 outputs 中的负类别（第一列）。

以下是如何编写代码来计算二元交叉熵损失的示例：
import torch
import torch.nn as nn

# 假设模型的输出，形状为 [batch_size, num_classes]
outputs = torch.tensor([[-0.4018, -1.0492],
                        [-0.7078,  0.0057],
                        [ 1.2464, -0.6230],
                        [ 0.1175, -0.0176]])

# 真实标签，形状为 [batch_size, 1]
targets = torch.tensor([[0],
                        [1],
                        [1],
                        [0]])

# 将targets的形状从 [batch_size, 1] 转换为 [batch_size]
# 并确保targets是长整型
targets = targets.squeeze(1).long()

# 选择outputs中与targets对应的类别得分
# 由于targets都是0或1，我们可以通过targets来索引outputs
# 例如，如果targets[i]是1，我们选择outputs[i][1]，反之选择outputs[i][0]
# 我们使用gather函数来实现这一点
logits = outputs.gather(1, targets.view(-1, 1)).squeeze(1)

# 创建二元交叉熵损失函数
loss_bce = nn.BCEWithLogitsLoss()

# 计算损失
bce_loss = loss_bce(logits, targets.float())  # 将targets转换为浮点型

print("二元交叉熵损失:", bce_loss.item())

在这个示例中，我们首先将 targets 从形状 [4, 1] 转换为形状 [4] 的一维张量，并确保它是长整型。然后，我们使用 outputs.gather() 方法根据 targets 选择正确的得分。gather() 方法根据 targets 中的索引从 outputs 中选择相应的得分。最后，我们使用 nn.BCEWithLogitsLoss 来计算二元交叉熵损失。注意，我们需要将 targets 转换为浮点型，因为 nn.BCEWithLogitsLoss 期望浮点型的目标值。
'''