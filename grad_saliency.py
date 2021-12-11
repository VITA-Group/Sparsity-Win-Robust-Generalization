# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from models.preactivate_resnet import *
from datasets import cifar10_dataloaders, cifar10_test_dataloaders
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
import torch.nn as nn
from  torchvision.utils import save_image 
import argparse
import os
import torch
import pdb
import numpy as np
import cv2
import tqdm
from skimage.io import imsave
from matplotlib import pyplot as plt
from advertorch.utils import NormalizeByChannelMeanStd
import numpy as np

from models.res import *
from models.preactivate_resnet import ResNet18

parser = argparse.ArgumentParser(description='Init Sparse Training Mask')
parser.add_argument('--model_dir', help='The directory load the trained models', default=None, type=str)

parser.add_argument('--out', help='The directory load the trained models', default=None, type=str)

parser.add_argument('--best_check', action='store_true', help='best checkpoint (default: off)')
#### adv ######
parser.add_argument('--train_eps', default=8, type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2, type=float, help='step size of attack during training')
parser.add_argument('--train_randinit', action='store_false', help='randinit usage flag (default: on)')
parser.add_argument('--test_eps', default=8, type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2, type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit', action='store_false', help='randinit usage flag (default: on)')

def saliency(img, model):

    for p in model.parameters():
        p.requires_grad = False 

    model.eval()

    img.unsqueeze_(0)
    img.requires_grad = True 

    output = model(img)
    max_index = output.argmax(dim=1)
    score = output[0, max_index]
    score.backward()

    return img.grad.data


def reshape_transform(tensor, height=7, width=7):
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == '__main__':
    args = parser.parse_args()
    args.train_eps = args.train_eps / 255
    args.train_gamma = args.train_gamma / 255
    args.test_eps = args.test_eps / 255
    args.test_gamma = args.test_gamma / 255

    dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    model = resnet18_eb(num_classes=10)
    model.normalize = dataset_normalization
    model = prune_unstructured(model, 1-0.1)

    # model = ResNet18(num_classes=10)
    # model.normalize = dataset_normalization

    print(args.model_dir)
    path = os.path.join(args.model_dir, 'checkpoint.pth.tar')
    if args.best_check:
        path = os.path.join(args.model_dir, 'model_RA_best.pth.tar')
    checkpoint = torch.load(path, map_location = 'cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()


    target_layer = model.layer4[-1]
    sample = 100
    test_loader = cifar10_test_dataloaders(batch_size = sample, data_dir = 'data/cifar10')
    (input_tensor, target) = next(iter(test_loader))
    input_tensor = input_tensor.cuda()
    target = target.cuda()

    save_dir = os.path.join(args.model_dir, 'final_32')
    if args.best_check:
        save_dir = os.path.join(args.model_dir, 'best_32')

    os.makedirs(save_dir, exist_ok=True)

    #adv samples
    criterion = nn.CrossEntropyLoss()

    adversary = LinfPGDAttack(
        model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
        rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
    )

    with ctx_noparamgrad(model):
        input_adv = adversary.perturb(input_tensor, target)
    

    all_sliency_map = []
    for img_id in range(sample):
        gradient = saliency(input_adv[img_id], model)
        all_sliency_map.append(gradient)

    all_sliency_map = torch.cat(all_sliency_map, dim=0)

    print(all_sliency_map.shape)

    torch.save({'image':input_adv, 'map':all_sliency_map}, args.out)

    print("finish!")




