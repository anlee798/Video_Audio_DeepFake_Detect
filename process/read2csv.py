import os
import csv


def process_labels(label_file, video_dir, audio_dir, output_csv):
    with open(label_file, 'r') as f:
        labels = f.readlines()

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 写入表头
        csvwriter.writerow(['video_path', 'audio_path', 'label'])
        for line in labels:
            video_file, label = line.strip().split(',')
            video_path = os.path.join(video_dir, video_file)
            # if not os.path.exists(video_path):
            #     print("视频文件", video_path, "不存在")
            audio_file = os.path.splitext(video_file)[0] + '.wav'
            audio_path = os.path.join(audio_dir, audio_file)
            # if not os.path.exists(audio_path):
            #     print("音频文件", audio_path, "不存在")
            # 写入一行数据
            if os.path.exists(video_path) and os.path.exists(audio_path):
                csvwriter.writerow([video_path, audio_path, label])
            else:
                print("video_path:", video_path, ", audio_path:", audio_path)


# 设置目录和文件路径
# base_dir = '/data2/zhuanlei/Datasets_DeepFake_Video_Audio/multi-ffdv/phase1'
base_dir = '/data/zhuanlei/phase1'
train_label_file = os.path.join(base_dir, 'trainset_label.txt')
val_label_file = os.path.join(base_dir, 'valset_label.txt')
train_video_dir = os.path.join(base_dir, 'trainset')
val_video_dir = os.path.join(base_dir, 'valset')
train_audio_dir = os.path.join(base_dir, 'wav', 'trainset')
val_audio_dir = os.path.join(base_dir, 'wav', 'valset')
train_output_csv = os.path.join(base_dir, 'train_list.csv')
val_output_csv = os.path.join(base_dir, 'val_list.csv')

# 生成 CSV 文件
process_labels(train_label_file, train_video_dir, train_audio_dir, train_output_csv)
process_labels(val_label_file, val_video_dir, val_audio_dir, val_output_csv)

print(f'{train_output_csv} 和 {val_output_csv} 生成成功。')
