from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms._transforms_video import NormalizeVideo
import torchaudio
import torch
from torchvision import transforms
import logging
import torch.nn.functional as F

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints

def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    fbank = fbank.transpose(0, 1)
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if abs(p) / n_frames > 0.2:
    #     logging.warning(
    #         "Large gap between audio n_frames(%d) and "
    #         "target_length (%d). Is the audio_target_length "
    #         "setting correct?",
    #         n_frames,
    #         target_length,
    #     )
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    fbank = fbank.unsqueeze(0)
    return fbank

def interpolate_frames(video_clip, target_fps, original_fps):
    num_target_frames = int(video_clip.shape[1] * (target_fps / original_fps))
    interpolated_clip = F.interpolate(video_clip.unsqueeze(0), size=(num_target_frames, video_clip.shape[2], video_clip.shape[3]), mode='trilinear', align_corners=False)
    return interpolated_clip.squeeze(0)

def load_video_audio_to_tensor(video_paths,
                               audio_paths,
                               device,
                               clip_duration=1,
                               clips_per_video=1,
                               sample_rate=16000,
                               num_mel_bins=128,
                               target_length=204,
                               mean=-4.268,
                               std=9.138,
                               target_video_fps=None,
                               get_target_video_method='indices', # indices or interpolate
                               ):
    '''
    :param video_paths: 视频位置数组
    :param audio_paths: 音频位置数组
    :param device: 设备
    :param clip_duration: 一个片段时间间隔 以秒为单位
    :param clips_per_video: 每个视频取多少个片段
    :param sample_rate: 16000 or 44100
    :param num_mel_bins:
    :param target_length:
    :param mean:
    :param std:
    :return: 返回 音视频处理后的Tensor  格式：视频：torch.Size([1, 1, 3, 25, 224, 224])
                                          音频：torch.Size([1, 1, 1, 128, 204])
    :param target_video_fps: target_video_fps 默认为None，即根据视频的fps决定最后输出T维度
                             target_video_fps为数值，则根据get_target_video_method进行扩展或缩进T维度
    :param get_target_video_method: 默认为 'indices',根据时间索引，部分的重复选取某些帧
                                        'interpolate'，根据线性插值的方式得到想要的帧数
    '''
    if (audio_paths is None) or (video_paths is None) :
        return None

    audio_outputs = []
    video_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    video_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            NormalizeVideo(
                mean=(0.45, 0.45, 0.45),
                std=(0.225, 0.225, 0.225)
            ),
        ]
    )

    for i, (video_path,audio_path) in enumerate(zip(video_paths,audio_paths)):
        # 音频处理
        waveform, sr = torchaudio.load(audio_path)
        video = EncodedVideo.from_path(video_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        clip_timepoints_duration = min(video.duration, waveform.size(1) / sample_rate)
        #TODO 参数2取相同 视频音频长度不一致问题解决办法
        # all_clips_timepoints = get_clip_timepoints(
        #     clip_sampler, waveform.size(1) / sample_rate
        # )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, clip_timepoints_duration
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                            :,
                            int(clip_timepoints[0] * sample_rate): int(
                                clip_timepoints[1] * sample_rate
                            ),
                            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        # all_clips = [normalize(ac).to(device) for ac in all_clips]
        all_clips = [normalize(ac) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

        # 视频处理
        # all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)
        #TODO 参数2取相同 视频音频长度不一致问题解决办法
        # all_clips_timepoints = get_clip_timepoints(clip_sampler, 3)

        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            video_clip = video.get_clip(start_sec=clip_timepoints[0], end_sec=clip_timepoints[1])['video']

            total_frames = video_clip.shape[1]
            original_fps = video_clip.shape[1] / (clip_timepoints[1] - clip_timepoints[0])
            if target_video_fps and original_fps != target_video_fps:
                if get_target_video_method == 'indices':
                    frame_indices = torch.linspace(0, total_frames - 1,
                                                   int(target_video_fps * (clip_timepoints[1] - clip_timepoints[0])))
                    frame_indices = frame_indices.round().long()
                    video_clip = video_clip[:, frame_indices, :, :]
                elif get_target_video_method == 'interpolate':
                    # 插帧以确保每秒提取30帧
                    video_clip = interpolate_frames(video_clip, target_video_fps,
                                                    original_fps=original_fps)  # torch.Size([3, 30, 384, 384])

            video_clip = video_transform(video_clip)
            all_clips.append(video_clip)

        video_outputs.append(torch.stack(all_clips, dim=0))
    return torch.stack(video_outputs, dim=0), torch.stack(audio_outputs, dim=0)
    # return torch.tensor(video_outputs), torch.tensor(audio_outputs)

# video_paths = ['output25fps.mp4']
# audio_paths = ['output25fps_16000.wav']
# video_tensor, audio_tensor = load_video_audio_to_tensor(video_paths,audio_paths,device='cpu')
# print(video_tensor.shape)
# print(audio_tensor.shape)
