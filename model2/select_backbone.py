## code source - https://github.com/TengdaHan/DPC/tree/master/backbone
from .resnet_2d3d import *
import torch
def select_resnet(network, track_running_stats=True,):
    param = {'feature_size': 1024}
    if network == 'resnet18':
        model = resnet18_2d3d_full(track_running_stats=track_running_stats)
        param['feature_size'] = 256
    elif network == 'resnet34':
        model = resnet34_2d3d_full(track_running_stats=track_running_stats)
        # 加载预训练模型 https://github.com/TengdaHan/DPC/blob/master/backbone/select_backbone.py
        pretrained_path = './weights/k400_224_r34_dpc-rnn_runningStats.pth.tar'
        print("=> loading pretrained checkpoint '{}'".format(pretrained_path))
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))['state_dict']
        model_state_dict = model.state_dict()
        for k, v in checkpoint.items():
            print("checkpoint k", k)
        for k, v in model_state_dict.items():
            print("model_state_dict k", k)
        for k, v in checkpoint.items():
            name = k.replace('module.backbone.', '')
            if name in model_state_dict:
                print("name:",name)
                model_state_dict[name] = v
        model.load_state_dict(model_state_dict)
        print("=> loading pretrained checkpoint '{}' Success!".format(pretrained_path))
        param['feature_size'] = 256 
    elif network == 'resnet50':
        model = resnet50_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet101':
        model = resnet101_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet152':
        model = resnet152_2d3d_full(track_running_stats=track_running_stats)
    elif network == 'resnet200':
        model = resnet200_2d3d_full(track_running_stats=track_running_stats)
    else: raise IOError('model type is wrong')

    return model, param