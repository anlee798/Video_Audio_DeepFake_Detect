# -*- coding: utf-8 -*-
# author: AnLee
import os
import subprocess

def convert_videos_to_30fps(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 可以根据需要添加其他视频格式
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            command = f"ffmpeg -i {input_path} -r 30 {output_path}"
            subprocess.run(command, shell=True)

input_folder = '/data/zhuanlei/phase1/trainset'
output_folder = '/data2/zhuanlei/Datasets_DeepFake_Video_Audio/multiFFDV-phase1/trainset'

convert_videos_to_30fps(input_folder, output_folder)

