# from model import Batfd
# import torch
#
# if __name__ == '__main__':
#     # video.shape torch.Size([16, 3, 512, 96, 96])
#     # audio.shape torch.Size([16, 64, 2048])
#     video_input = torch.randn(2, 3, 512, 96, 96)
#     audio_input = torch.randn(2, 64, 2048)
#     model = Batfd()
#     print(model)
#     output = model(video_input, audio_input)
#     print(output) # torch.Size([1, 1])

from model2 import Audio_RNN
import torch

if __name__ == '__main__':
    video_input = torch.randn(4, 1, 3, 30, 224, 224)
    audio_input = torch.randn(4, 1, 1, 13, 99)
    model = Audio_RNN(img_dim=224, network='resnet34')
    # output = model(video_input, audio_input)
    # print(output.shape)

    from thop import clever_format, profile
    flops, params = profile(model.to('cpu'), (video_input.to('cpu'), audio_input.to('cpu')))
    print(f'Total GFLOPS: {flops / 1e9} Billion')   # Total GFLOPS: 116.513 Billion
    print(f'Total params: {clever_format(params)}') # Total params: 142.19M