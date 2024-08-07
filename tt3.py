# -*- coding: utf-8 -*-
# author: AnLee
import os
folder_path = '/data2/zhuanlei/Datasets_DeepFake_Video_Audio/multi-ffdv/phase1/wav/trainset'
filename = '/data2/zhuanlei/Datasets_DeepFake_Video_Audio/multi-ffdv/phase1/wav/trainset/239158f052174d358409e7865bea8779.wav'
if os.path.exists(filename):
    print("文件", filename, "存在")
else:
    print("不存在")

if os.path.isdir(folder_path):
    print("文件夹存在")
else:
    print("文件夹不存在")