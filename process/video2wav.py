import os
import subprocess


def extract_audio_from_video(video_path, audio_path):
    # 使用 ffmpeg 提取音频
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '2',
        audio_path
    ]
    subprocess.run(command, check=True)


def process_videos(video_dir, audio_dir):
    # 如果音频目录不存在，则创建
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # 遍历视频目录中的所有文件
    for filename in os.listdir(video_dir):
        video_path = os.path.join(video_dir, filename)

        # 只处理视频文件
        if os.path.isfile(video_path) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # 确定音频文件的路径和名称
            audio_filename = os.path.splitext(filename)[0] + '.wav'
            audio_path = os.path.join(audio_dir, audio_filename)

            # 提取音频
            extract_audio_from_video(video_path, audio_path)
            print(f'提取音频: {video_path} -> {audio_path}')


# 处理 trainset 和 valset 文件夹
# base_dir = 'mydeepfakedata'
base_dir = '/data/zhuanlei/phase1'
trainset_video_dir = os.path.join(base_dir, 'trainset')
valset_video_dir = os.path.join(base_dir, 'valset')
trainset_audio_dir = os.path.join(base_dir, 'wav', 'trainset')
valset_audio_dir = os.path.join(base_dir, 'wav', 'valset')

process_videos(trainset_video_dir, trainset_audio_dir)
process_videos(valset_video_dir, valset_audio_dir)
