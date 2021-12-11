'''
Adversarial Training 

'''
import os
import sys
from numpy.core.numeric import outer 
import torch
import pickle
import argparse
import torch.optim
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.models as models
from utils import *
from sparselearning.core import Masking, CosineDecay
from sparselearning.pruning_utils import check_sparsity


parser = argparse.ArgumentParser(description='PyTorch Adversarial Sparse Training')

########################## data setting ##########################
parser.add_argument('--data', type=str, default='data/cifar10', help='location of the data corpus', required=True)
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset [cifar10, cifar100, tinyimagenet]', required=True)

########################## model setting ##########################
parser.add_argument('--arch', type=str, default='resnet18', help='model architecture [resnet18, wideresnet, vgg16]', required=True)
parser.add_argument('--depth_factor', default=34, type=int, help='depth-factor of wideresnet')
parser.add_argument('--width_factor', default=10, type=int, help='width-factor of wideresnet')

########################## basic setting ##########################
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--resume_dir', help='The directory resume the trained models', default=None, type=str)
parser.add_argument('--pretrained', default=None, type=str, help='pretrained model')
parser.add_argument('--eval', action="store_true", help="evaluation pretrained model")
parser.add_argument('--print_freq', default=50, type=int, help='logging frequency during training')
parser.add_argument('--save_dir', help='The parent directory used to save the trained models', default=None, type=str)

########################## training setting ##########################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--decreasing_lr', default='100,150', help='decreasing strategy')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')

########################## attack setting ##########################
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--train_eps', default=8, type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2, type=float, help='step size of attack during training')
parser.add_argument('--train_randinit', action='store_false', help='randinit usage flag (default: on)')
parser.add_argument('--test_eps', default=8, type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2, type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit', action='store_false', help='randinit usage flag (default: on)')

########################## Flying Bird setting ##########################
parser.add_argument('--fb', action='store_true', help='Enable flying bird mode. Default: True.')
parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
parser.add_argument('--sparse_init', type=str, default='igq', help='sparse initialization')
parser.add_argument('--reset', action='store_true', help='Fix topology during training. Default: True.')
parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.')
parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
parser.add_argument('--density', type=float, default=0.2, help='The density of the overall sparse network.')
parser.add_argument('--dynamic_epoch', default=100, type=int, help='The dynamic sparse start epoch')
parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
parser.add_argument('--update_frequency', type=int, default=900, metavar='N', help='how many iterations to train between mask update')


########################## Flying Bird+ sparse ##########################
parser.add_argument('--fbp', action='store_true', help='Enable fling bird+ mode. Default: True.')
parser.add_argument('--epoch_range', type=int, default=4, help='epoch range to decide sparse action')
parser.add_argument('--prune_ratio', type=float, default=0.4, help='The ratio of dynamic prune ')
parser.add_argument('--growth_ratio', type=float, default=0.05, help='The ratio of dynamic growth ')
parser.add_argument('--update_threshold', type=float, default=0.5, help='The update threshold of dynamic prune or growth')

########################## Dynamic frequency #############################
parser.add_argument('--dynamic_fre', action='store_true', help='Enable dynamic frequency mode. Default: True.')
parser.add_argument('--second_frequency', type=int, default=1200, metavar='N', help='how many iterations to train between mask update in second stage')




def main():

    args = parser.parse_args()
    args.train_eps = args.train_eps / 255
    args.train_gamma = args.train_gamma / 255
    args.test_eps = args.test_eps / 255
    args.test_gamma = args.test_gamma / 255
    
    save_path = get_save_path(args)
    os.makedirs(save_path, exist_ok=True)
    setup_logger(args)
    print_args(args)
    print_and_log(args)


    torch.cuda.set_device(int(args.gpu))

    if args.seed:
        print('set random seed = ', args.seed)
        setup_seed(args.seed)

    train_loader, val_loader, test_loader, model = setup_dataset_models(args)

    model.cuda()

    ########################## optimizer and scheduler ##########################
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr) for lr in args.decreasing_lr.split(',')], last_epoch=-1)



    ########################## sparse  mask ####################################
    mask = None
    if args.fb or args.fbp:
        decay = CosineDecay(args.death_rate, len(train_loader) * (args.epochs))
        mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                        redistribution_mode=args.redistribution, args=args)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density)
        # mask.set_dst_start_epoch(0.5*args.epochs)
        mask.set_dst_start_epoch(args.dynamic_epoch)

    ######################### only evaluation ###################################
    if args.eval:
        assert args.pretrained
        pretrained_model = torch.load(args.pretrained, map_location = torch.device('cuda:'+str(args.gpu)))
        print_and_log('loading from state_dict')
        if 'state_dict' in pretrained_model.keys():
            pretrained_model = pretrained_model['state_dict']
        model.load_state_dict(pretrained_model)
        test(test_loader, model, criterion, args)
        test_adv(test_loader, model, criterion, args)
        return

    ########################## resume ##########################
    start_epoch = 0
    if args.resume:
        print_and_log('resume from checkpoint.pth.tar')
        checkpoint = torch.load(os.path.join(args.resume_dir, 'checkpoint.pth.tar'), map_location = torch.device('cuda:'+str(args.gpu)))
        best_sa = checkpoint['best_sa']
        best_ra = checkpoint['best_ra']
        best_test_sa = checkpoint['best_test_sa']
        best_test_ra = checkpoint['best_test_ra']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        all_result = checkpoint['result']
        if mask:
            mask.load_info_resume(checkpoint['mask_info'])
        
        set_ite_step(checkpoint['iter_step'])
    else:
        all_result = {}
        all_result['train_acc'] = []
        all_result['val_sa'] = []
        all_result['val_ra'] = []
        all_result['test_sa'] = []
        all_result['test_ra'] = []
        all_result['g_norm'] = []
        all_result['rc_ratio'] = []
        all_result['grad_cs'] = []
        all_result['sparsity'] = 1.0
        all_result['total_fired_weights'] = 0
        all_result['best_ra_epoch'] = 0
        best_sa = 0
        best_ra = 0
        best_test_sa = 0
        best_test_ra = 0

    is_sa_best = False
    is_ra_best = False
    is_test_sa_best = False
    is_test_ra_best = False

    ########################## training process ##########################
    for epoch in range(start_epoch, args.epochs):

        current_sparsity = check_sparsity(model)
        print_and_log(optimizer.state_dict()['param_groups'][0]['lr'])

        if mask:
            mask.set_dst_current_epoch(epoch)

        print_and_log('baseline adversarial training')
        train_acc = train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args, mask)

        all_result['train_acc'].append(train_acc)
        scheduler.step()
        all_result['sparsity'] = current_sparsity

        ###validation###
        val_sa = test(val_loader, model, criterion, args)
        val_ra, val_loss = test_adv(val_loader, model, criterion, args)   
        test_sa = test(test_loader, model, criterion, args)
        test_ra, _= test_adv(test_loader, model, criterion, args)  

        if args.fbp:
            mask.update_loss_info(val_loss)
            mask.update_train_val_diff(train_acc, val_ra)

        all_result['val_sa'].append(val_sa)
        all_result['val_ra'].append(val_ra)
        all_result['test_sa'].append(test_sa)
        all_result['test_ra'].append(test_ra)

        is_sa_best = val_sa  > best_sa
        best_sa = max(val_sa, best_sa)

        is_ra_best = val_ra  > best_ra
        best_ra = max(val_ra, best_ra)

        if is_ra_best:
            all_result['best_ra_epoch'] = epoch

        checkpoint_state = {
            'best_sa': best_sa,
            'best_ra': best_ra,
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iter_step': get_ite_step(),
            'result': all_result
        }

        if mask:
            checkpoint_state.update({
                'mask_info': mask.get_info_resume()
            })
        
        if args.fbp:
            prune_epochs, growth_epochs = mask.get_dst_epochs()
            all_result['prune_epochs'] = prune_epochs
            all_result['growth_epochs'] = growth_epochs

        if mask:
            all_result['total_fired_weights']  = mask.total_fired_weights

        checkpoint_state.update({
            'result': all_result
        })
        save_checkpoint(checkpoint_state, is_sa_best, is_ra_best, save_path)

        if epoch == args.epochs - 1:
            getinfo(checkpoint_state)
            get_generalization_gap(model, criterion, args)

        plt.plot(all_result['train_acc'], label='train_acc')
        plt.plot(all_result['test_sa'], label='SA')
        plt.plot(all_result['test_ra'], label='RA')
        
        if args.fbp :
            prune_epochs, growth_epochs = mask.get_dst_epochs()
            delta = 0.4
            for epoch in prune_epochs:
                plt.axvline(epoch - delta, linewidth = 0.8, color = 'black', linestyle='--')
            
            for epoch in growth_epochs:
                plt.axvline(epoch + delta, linewidth = 0.8, color = 'red', linestyle='--')


        plt.legend()
        plt.savefig(os.path.join(save_path, 'net_train.png'))
        plt.close()


if __name__ == '__main__':
    main()


