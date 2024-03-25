# import sys
from cnn import cnn
from vgg import vgg16, vgg19
from configs import config_args
from torch import nn
import torch
import numpy as np

# sys.path.append("..")
# 等价于 sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# @ profile
def create_model(dataset):
    model = None
    if dataset in ['mnist', 'fmnist']:
        model = nn.Sequential(list(cnn().children())[0])
    elif dataset == 'cifar10':
        model = nn.Sequential(list(vgg16().children())[0])
    elif dataset == 'cifar100':
        model = nn.Sequential(list(vgg19().children())[0])
    return model.to(config_args.device)


if __name__ == '__main__':
    model = create_model(config_args.dataset).cpu()
    children = list(model.children())
    # [Sequential(
    #   (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
    #   (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     
    #   (2): ReLU()
    #   (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    #   (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
    #   (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     
    #   (6): ReLU()
    #   (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    #   (8): FlattenLayer()
    #   (9): Linear(in_features=1024, out_features=512, bias=True)
    #   (10): ReLU()
    #   (11): Linear(in_features=512, out_features=10, bias=True)
    # )]
    print(children)
    for name, param in model.state_dict().items():
        # 0.0.weight
        # 0.0.bias
        # 0.1.weight
        # 0.1.bias
        # 0.1.running_mean
        # 0.1.running_var
        # 0.1.num_batches_tracked
        # 0.4.weight
        # 0.4.bias
        # 0.5.weight
        # 0.5.bias
        # 0.5.running_mean
        # 0.5.running_var
        # 0.5.num_batches_tracked
        # 0.9.weight
        # 0.9.bias
        # 0.11.weight
        # 0.11.bias
        print(f'{name}')
    model = children[0]
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    # Model Sequential : params: 0.199463M
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1024 / 1024))
    # '''
    from thop import profile
    input = torch.randn(64, 1, 28, 28)
    flops, params = profile(model, inputs=(input,))
    # 0.052288 0.244973568 
    print(params / 1e6, flops / 1e9)  # flops单位G，para单位M
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("---|---|---")
    # cnn | 0.05 | 0.24
    print("%s | %.2f | %.2f" % ('cnn', params / (1000 ** 2), flops / (1000 ** 3)))

    import time
    import os
    localtime = time.localtime(time.time())
    path = f"{config_args.protocol}_{localtime[1]:02}{localtime[2]:02}{localtime[3]:02}{localtime[4]:02}"
    saved_model_path = os.path.join('../saved_models', path)
    saved_results_path = os.path.join('../results', config_args.protocol)
    res_path = os.path.join(saved_model_path, f"{config_args.dataset}_{path}.pt")
    arg_path = os.path.join(saved_results_path, f"{config_args.dataset}_{path}.txt")
    # ../saved_models\FedRich_03242141\mnist_FedRich_03242141.pt ../results\FedRich\mnist_FedRich_03242141.txt
    print(res_path, arg_path)
    # '''