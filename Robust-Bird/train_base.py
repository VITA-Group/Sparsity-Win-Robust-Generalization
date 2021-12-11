import os
import torch
import argparse
import torch.optim
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from advertorch.utils import NormalizeByChannelMeanStd
import numpy as np

from models import vgg16_bn, WideResNet, resnet18
from datasets import cifar10_dataloaders, cifar100_dataloaders, tiny_imagenet_dataloaders
from utils import train_epoch, train_epoch_adv, train_epoch_fast, test, test_adv, save_checkpoint, prune_unstructured


# ########################################## Hyperparameters ##########################################

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')

# experiment setting
parser.add_argument('--stage1', type=str, choices=['pgd','fast','sgd'], )
parser.add_argument('--epochs1', type=int, default=200)
parser.add_argument('--stage2', type=str, choices=['pgd','fast','sgd'], required=True)
parser.add_argument('--epochs2', type=int, required=True)
parser.add_argument('--stage1_pretrained', type=str, default=None, help='path to pretrained stage1 model')
parser.add_argument('--arch', type=str, choices=['resnet18', 'wideresnet', 'vgg16_bn'], required=True)
parser.add_argument('--pruning', type=str, choices=['unstructured','channel'], required=True)
parser.add_argument('--density', type=float, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--print_freq', type=int, default=100, help='print freq')

# robust overfitting setting
# training
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--decreasing_lr', default='50,150', help='decreasing strategy')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
# attack
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--train_eps', default=8, type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2, type=float, help='step size of attack during training')
parser.add_argument('--train_randinit', action='store_false', help='randinit usage flag (default: on)')
parser.add_argument('--test_eps', default=8, type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2, type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit', action='store_false', help='randinit usage flag (default: on)')
# wideResNet
parser.add_argument('--depth_factor', default=34, type=int, help='depth-factor of wideresnet')
parser.add_argument('--width_factor', default=10, type=int, help='width-factor of wideresnet')


def main():
    # ########################## Hyperparameters ##########################
    args = parser.parse_args()
    args.train_eps = args.train_eps / 255
    args.train_gamma = args.train_gamma / 255
    args.test_eps = args.test_eps / 255
    args.test_gamma = args.test_gamma / 255
    args.save_dir = 'store/'+args.save_dir

    torch.cuda.set_device(0)

    os.makedirs(f'{args.save_dir}_stage1', exist_ok=True)
    os.makedirs(f'{args.save_dir}_stage2', exist_ok=True)

    ########################################## dataset ##########################################
    if 'cifar10' in args.data and 'cifar100' not in args.data:
        n_classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    elif 'cifar100' in args.data:
        n_classes = 100
        dataset_normalization = NormalizeByChannelMeanStd(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    elif 'tiny_imagenet' in args.data:
        n_classes = 200
        dataset_normalization = NormalizeByChannelMeanStd(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size=args.batch_size, data_dir=args.data)


    # ########################################## model ##########################################
    if args.arch == 'resnet18':
        model = resnet18(num_classes=n_classes)
        model.normalize = dataset_normalization
        model = model.cuda()

    if args.arch == 'wideresnet':
        model = WideResNet(args.depth_factor, n_classes, widen_factor=args.width_factor, dropRate=0.0)
        model.normalize = dataset_normalization
        model = model.cuda()

    if args.arch == 'vgg16_bn':
        model = vgg16_bn(num_classes=n_classes)
        model.normalize = dataset_normalization
        model = model.cuda()


    # ############################################################################################
    # ################################# stage 1 (SGD by default) #################################
    # ############################################################################################

    ########################## optimizer and scheduler ##########################
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    ########################## store result ##########################
    all_result = {
        'train_acc': [],
        'val_sa': [],
        'val_ra': [],
        'test_sa': [],
        'test_ra': []
    }
    best_sa, best_ra = 0, 0

    for epoch in range(args.epochs1):
        print('STAGE 1 TRAINING')
        print(f'epoch {epoch}')
        print(f"lr {optimizer.state_dict()['param_groups'][0]['lr']}")

        # # ######################## Training ########################
        if args.stage1 == 'pgd':
            train_acc = train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args)
        elif args.stage1 == 'sgd':
            train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, args)
        elif args.stage1 == 'fast':
            train_acc = train_epoch_fast(train_loader, model, criterion, optimizer, epoch, 'stage1', args)

        all_result['train_acc'].append(train_acc)
        scheduler.step()

        # ######################## Evaluation ########################
        val_sa = test(val_loader, model, criterion, args)
        val_ra = test_adv(val_loader, model, criterion, args)   
        test_sa = test(test_loader, model, criterion, args)
        test_ra = test_adv(test_loader, model, criterion, args)  

        all_result['val_sa'].append(val_sa)
        all_result['val_ra'].append(val_ra)
        all_result['test_sa'].append(test_sa)
        all_result['test_ra'].append(test_ra)

        is_sa_best = val_sa  > best_sa
        best_sa = max(val_sa, best_sa)

        is_ra_best = val_ra  > best_ra
        best_ra = max(val_ra, best_ra)

        checkpoint_state = {
            'best_sa': best_sa,
            'best_ra': best_ra,
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'result': all_result
        }

        save_checkpoint(checkpoint_state, is_sa_best, is_ra_best, args.save_dir+'_stage1')

        plt.plot(all_result['train_acc'], label='train_acc')
        plt.plot(all_result['test_sa'], label='SA')
        plt.plot(all_result['test_ra'], label='RA')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir+'_stage1', 'net_train.png'))
        plt.close()

    # ######################### pruning ##########################
    if args.pruning == 'unstructured':
        model = prune_unstructured(model, 1-args.density)

    # ############################################################################################
    # ################################# stage 2 (SGD by default) #################################
    # ############################################################################################

    ########################## RESET optimizer and scheduler ##########################
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr,momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    ########################## RESET store result ##########################
    all_result = {
        'train_acc': [],
        'val_sa': [],
        'val_ra': [],
        'test_sa': [],
        'test_ra': []
    }
    best_sa, best_ra = 0, 0

    for epoch in range(args.epochs2):
        print('STAGE 2 TRAINING')
        print(f'epoch {epoch}')
        print(f"lr {optimizer.state_dict()['param_groups'][0]['lr']}")

        # ######################## Training ########################
        if args.stage2 == 'pgd':
            train_acc = train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args)
        elif args.stage2 == 'sgd':
            train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, args)
        elif args.stage2 == 'fast':
            train_acc = train_epoch_fast(train_loader, model, criterion, optimizer, epoch, 'stage2', args)

        all_result['train_acc'].append(train_acc)
        scheduler.step()

        # ######################## Evaluation ########################
        val_sa = test(val_loader, model, criterion, args)
        val_ra = test_adv(val_loader, model, criterion, args)   
        test_sa = test(test_loader, model, criterion, args)
        test_ra = test_adv(test_loader, model, criterion, args)  

        all_result['val_sa'].append(val_sa)
        all_result['val_ra'].append(val_ra)
        all_result['test_sa'].append(test_sa)
        all_result['test_ra'].append(test_ra)

        is_sa_best = val_sa  > best_sa
        best_sa = max(val_sa, best_sa)

        is_ra_best = val_ra  > best_ra
        best_ra = max(val_ra, best_ra)

        checkpoint_state = {
            'best_sa': best_sa,
            'best_ra': best_ra,
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'result': all_result
        }

        save_checkpoint(checkpoint_state, is_sa_best, is_ra_best, args.save_dir+'_stage2')

        plt.plot(all_result['train_acc'], label='train_acc')
        plt.plot(all_result['test_sa'], label='SA')
        plt.plot(all_result['test_ra'], label='RA')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir+'_stage2', 'net_train.png'))
        plt.close()

    
    # ################################################ Log Writing ################################################
    with open(args.save_dir+'_stage2/results.txt','a') as f:
        ra_best = np.max(all_result['test_ra'])
        ra_final = all_result['test_ra'][-1]
        ra_dif = ra_best - ra_final

        sa_best = np.max(all_result['test_sa'])
        sa_final = all_result['test_sa'][-1]
        sa_dif = sa_best - sa_final

        # training RA v.s. test RA
        gap_best = np.max(all_result['train_acc']) - ra_best # difference of RA training and RA testing (best)
        gap_final = all_result['train_acc'][-1] - ra_final # difference of RA training and RA testing (final)
        gap_dif = gap_best - gap_final

        # save file
        f.write(f'ra_best {ra_best}\n')
        f.write(f'ra_final {ra_final}\n')
        f.write(f'ra_dif {ra_dif}\n')
        f.write(f'sa_best {sa_best}\n')
        f.write(f'sa_final {sa_final}\n')
        f.write(f'sa_dif {sa_dif}\n')
        f.write(f'gap_best {gap_best}\n')
        f.write(f'gap_final {gap_final}\n')
        f.write(f'gap_dif {gap_dif}\n')


if __name__ == '__main__':
    main()