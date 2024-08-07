# -*- coding: utf-8 -*-
# author: AnLee
import os
import subprocess

# 指定视频文件所在的文件夹路径
video_folder_path = '/data/zhuanlei/phase1/trainset'
# 指定输出文本文件的路径
output_txt_path = '/data2/zhuanlei/Video_Audio_DeepFake_Detect/trainset_fps_list.txt'

# 支持的视频文件扩展名列表
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

# 确保输出文件路径存在
os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

# 遍历文件夹中的所有文件
with open(output_txt_path, 'w') as output_file:  # 打开文件用于写入
    for file_name in os.listdir(video_folder_path):
        # 检查文件扩展名是否为视频文件
        if file_name.lower().endswith(video_extensions):
            file_path = os.path.join(video_folder_path, file_name)

            # 调用ffprobe命令获取帧率
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of',
                     'default=noprint_wrappers=1', file_path],
                    stdout=subprocess.PIPE,  # 捕获输出
                    stderr=subprocess.STDOUT,  # 将错误重定向到输出
                    text=True  # 获取文本输出而不是二进制
                )

                # 检查ffprobe命令是否成功执行
                if result.returncode == 0:
                    fps = result.stdout.strip()
                    output_file.write(f"{file_name}: {fps}\n")  # 写入文件
                else:
                    print(f"无法获取 {file_name} 的帧率: ffprobe命令执行失败")
            except subprocess.CalledProcessError as e:
                print(f"无法获取 {file_name} 的帧率: {e}")

print(f"视频帧率信息已保存到 '{output_txt_path}'")
