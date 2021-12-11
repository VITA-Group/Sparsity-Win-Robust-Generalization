# from utils import *
import torch

import os
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from thop import profile
from ptflops import get_model_complexity_info

from advertorch.utils import NormalizeByChannelMeanStd
import numpy as np

# from utils import *
import argparse

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')

#experiment setting
parser.add_argument('--stage1_epochs', type=int, required=True)
parser.add_argument('--stage2_epochs', type=int, required=True)
parser.add_argument('--folder', type=str, required=True)


def main():

    args = parser.parse_args()

    # --------- get experiment setting from folder name -------------
    tokens = args.folder.split('_')
    t = tokens[3].split('+')
    args.stage1 = 'fast' if 'fast' in t[0] else 'sgd' if 'sgd' in t[0] else 'pgd' if 'pgd' in t[0] else None
    args.stage2 = 'fast' if 'fast' in t[1] else 'sgd' if 'sgd' in t[1] else 'pgd' if 'pgd' in t[1] else None
    args.density = float(tokens[4])

    n_classes = 100 if 'c100' in args.folder else 10 if 'c10' in args.folder else None

    dense = resnet18(seed=0, num_classes=n_classes).cuda()
    dataset_normalization = NormalizeByChannelMeanStd(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    dense.normalize = dataset_normalization
    flops_stage1 = flops(dense, args.stage1, args.stage1_epochs)

    sparse = resnet18(seed=0, num_classes=n_classes).cuda()
    dataset_normalization = NormalizeByChannelMeanStd(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    sparse.normalize = dataset_normalization
    sparse = prune_unstructured(sparse, 1-args.density)
    flops_stage2 = flops(sparse, args.stage2, args.stage2_epochs) * args.density # 0 multiplications dont count
    
    print(f'flops_stage1 = {flops_stage1}')
    print(f'flops_stage2 = {flops_stage2}')
    print(f'total flops = {flops_stage1+flops_stage2}')

    with open(f'{args.folder}/flops.txt', 'w') as f:
        f.write(f'flops_stage1 = {flops_stage1}\n')
        f.write(f'flops_stage2 = {flops_stage2}\n')
        f.write(f'total flops = {flops_stage1+flops_stage2}\n')


def flops(model, mode, epochs):

    assert mode in ['pgd','sgd','fast']

    input_ = torch.randn(1, 3, 32, 32).cuda()
    mac_per_img, n_params = profile(model, inputs=(input_,))
    flops_per_img = mac_per_img*2

    f1 = flops_per_img # 1 FP
    f2 = 3 * flops_per_img # 1 BP
    f3 = 3 * flops_per_img - 2*n_params + 2 * 1 * 32**2 # generate 1 adv img

    if mode == 'pgd':
        flops = f1+f2+10*f1+10*f3
    elif mode == 'fast':
        flops = f1+f2+f1+f3
    elif mode == 'sgd':
        flops = f1+f2

    flops_it = flops * 128 # batch size 128
    flops_epoch = flops_it * (50000 // 128)

    return epochs * flops_epoch

if __name__ == '__main__':
    main()





