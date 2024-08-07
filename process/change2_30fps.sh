#!/bin/bash

# 源文件夹路径
SOURCE_FOLDER="/data/zhuanlei/phase1/trainset"
# 目标文件夹路径
TARGET_FOLDER="/data2/zhuanlei/Datasets_DeepFake_Video_Audio/multiFFDV-phase1/trainset"
# 支持的视频文件扩展名列表
VIDEO_EXTENSIONS="(*.mp4 *.avi *.mov *.mkv)"

# 创建目标文件夹如果它不存在
mkdir -p "${TARGET_FOLDER}"

# 遍历源文件夹中的所有视频文件
find "${SOURCE_FOLDER}" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) | while read video_file; do
    # 从源文件路径中提取文件名
    filename=$(basename -- "$video_file")
    # 构建目标文件路径
    target_file="${TARGET_FOLDER}/${filename}"

    # 使用ffmpeg转换视频帧率
    ffmpeg -i "$video_file" -vf "fps=30" -c:v libx264 -c:a copy "$target_file"
done

echo "视频转换完成，已保存到 '${TARGET_FOLDER}'"