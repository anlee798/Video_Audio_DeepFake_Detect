# -*- coding: utf-8 -*-
# author: AnLee
import os

# 视频文件夹路径
video_train_folder_path = '/data/zhuanlei/phase1/trainset'
# video_val_folder_path = '/data/zhuanlei/Datasets_DeepFake_Video_Audio/multi-ffdv/phase1/valset'
# 标签文件路径
labels_file_path = '/data/zhuanlei/phase1/trainset_label.txt'

# 用于存储结果的字典
results = {}

# 读取标签文件
with open(labels_file_path, 'r') as file:
    for line in file:
        # 假设每行的格式是 "视频文件名.mp4,标签"
        video_filename, label = line.strip().split(',')
        video_train_filepath = os.path.join(video_train_folder_path, video_filename)
        # video_val_filepath = os.path.join(video_val_folder_path, video_filename)
        # print("video_train_filepath",video_train_filepath)
        # print("video_val_filepath",video_val_filepath)

        # print(video_filepath)

        # 检查视频文件是否存在
        # if not (os.path.exists(video_train_filepath) or os.path.exists(video_val_filepath)):
        #     results[video_filename] = ('存在', label)
        if not os.path.exists(video_train_filepath):
            results[video_filename] = ('不存在', label)

# 打印结果
# for video_filename, status in results.items():
#     print(f"视频文件 {video_filename} - 状态: {status[0]}, 标签: {status[1]}")

print("视频文件不存在数量:", len(results))

# import os
#
# # 视频文件夹路径
# # video_folder_path = '/data2/zhuanlei/Datasets_DeepFake_Video_Audio/multi-ffdv/phase1/trainset'
# video_folder_path = '/data/zhuanlei/phase1/trainset'
# # 要保存视频文件名的文本文件路径
# output_file_path = '/data2/zhuanlei/Video_Audio_DeepFake_Detect/new_trainset.txt'
#
# # 支持的视频文件扩展名列表
# video_extensions = ('.mp4', '.avi', '.mov', '.mkv')  # 根据需要添加或删除扩展名
#
# # 遍历视频文件夹，收集所有视频文件名
# video_files = [f for f in os.listdir(video_folder_path)
#                if os.path.isfile(os.path.join(video_folder_path, f))
#                and f.lower().endswith(video_extensions)]
#
# # 将视频文件名按字母顺序排序（可选）
# video_files.sort()
#
# # 将视频文件名写入到文本文件中
# with open(output_file_path, 'w') as output_file:
#     for video_file in video_files:
#         output_file.write(video_file + '\n')
#
# print(f"视频文件名已保存到 '{output_file_path}'")
# print(len(video_files))