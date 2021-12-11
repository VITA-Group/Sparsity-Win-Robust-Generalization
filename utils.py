import os
import time
from numpy.lib.function_base import gradient 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
from advertorch.utils import NormalizeByChannelMeanStd
from datasets import *
from models.preactivate_resnet import *
from models.vgg import *
from models.wideresnet import *
import hashlib
import logging
from sparselearning.pruning_utils import check_sparsity

__all__ = ['save_checkpoint', 'setup_dataset_models', 'setup_seed', 'print_args', 'train_epoch_adv', 
            'get_ite_step', 'set_ite_step', 'get_generalization_gap', 'test', 'test_adv', 
            'get_save_path', 'setup_logger', 'print_and_log', 'generate_adv', 'getinfo']

logger = None
def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)
    
    save_path = get_save_path(args)

    log_path = os.path.join(save_path, 'result.log')

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    if logger:
        logger.info(msg)

def save_checkpoint(state, is_SA_best, is_RA_best, save_path, filename='checkpoint.pth.tar'):
    
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)

    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))
    if is_RA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_RA_best.pth.tar'))

#print training configuration
def print_args(args):
    print('*'*50)
    print('Dataset: {}'.format(args.dataset))
    print('Model: {}'.format(args.arch))
    if args.arch == 'wideresnet':
        print('Depth {}'.format(args.depth_factor))
        print('Width {}'.format(args.width_factor))
    print('*'*50)        
    print('Attack Norm {}'.format(args.norm))  
    print('Test Epsilon {}'.format(args.test_eps))
    print('Test Steps {}'.format(args.test_step))
    print('Train Steps Size {}'.format(args.test_gamma))
    print('Test Randinit {}'.format(args.test_randinit))
    if args.eval:
        print('Evaluation')
        print('Loading weight {}'.format(args.pretrained))
    else:
        print('Training')
        print('Train Epsilon {}'.format(args.train_eps))
        print('Train Steps {}'.format(args.train_step))
        print('Train Steps Size {}'.format(args.train_gamma))
        print('Train Randinit {}'.format(args.train_randinit))

def setup_dataset_models_eval(args):

    # prepare dataset
    if args.dataset == 'cifar10':
        classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders_eval(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'cifar100':
        classes = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_loader, val_loader, test_loader = cifar100_dataloaders_eval(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'tinyimagenet':
        classes = 200
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_loader, val_loader, test_loader = tiny_imagenet_dataloaders_eval(batch_size = args.batch_size, data_dir = args.data)
    
    else:
        raise ValueError("Unknown Dataset")

    #prepare model

    if args.arch == 'resnet18':
        model = ResNet18(num_classes = classes)
        model.normalize = dataset_normalization

    elif args.arch == 'wideresnet':
        model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
        model.normalize = dataset_normalization

    elif args.arch == 'vgg16':
        model = vgg16_bn(num_classes = classes)
        model.normalize = dataset_normalization

    else:
        raise ValueError("Unknown Model")
    
    return train_loader, val_loader, test_loader, model

# prepare dataset and models
def setup_dataset_models(args):

    # prepare dataset
    if args.dataset == 'cifar10':
        classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'cifar100':
        classes = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'tinyimagenet':
        classes = 200
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    else:
        raise ValueError("Unknown Dataset")

    #prepare model

    if args.arch == 'resnet18':
        model = ResNet18(num_classes = classes)
        model.normalize = dataset_normalization

    elif args.arch == 'wideresnet':
        model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
        model.normalize = dataset_normalization

    elif args.arch == 'vgg16':
        model = vgg16_bn(num_classes = classes)
        model.normalize = dataset_normalization

    else:
        raise ValueError("Unknown Model")
    
    return train_loader, val_loader, test_loader, model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

ite_step = 0

def get_ite_step():
    global ite_step
    return ite_step

def set_ite_step(step):
    global ite_step
    print_and_log("set ite_step: {}".format(step))
    ite_step = step

def train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args, mask):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )

    model.train()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        with ctx_noparamgrad(model):
            input_adv = adversary.perturb(input, target)

        # compute output
        output_adv = model(input_adv)
        loss = criterion(output_adv, target)

        global ite_step

        optimizer.zero_grad()
        loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print_and_log('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()
        
        
        # update sparse topology
        # global ite_step
        update_frequency = args.update_frequency
        if args.dynamic_fre and epoch > (args.epochs / 2):
            update_frequency = args.second_frequency

        ite_step += 1
        if (args.fb or args.fbp) and ite_step % update_frequency == 0 and not args.fix:
            mask.at_end_of_epoch()

    print_and_log('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

#testing
def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Standard Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test_adv(val_loader, model, criterion, args):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        input_adv = adversary.perturb(input, target)
        # compute output
        with torch.no_grad():
            output = model(input_adv)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print_and_log('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print_and_log('Robust Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg



def generate_adv(val_loader, model, criterion, args):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.eval()
    start = time.time()

    all_adv_image = []
    all_target = []

    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        input_adv = adversary.perturb(input, target)
        # compute output
        with torch.no_grad():
            output = model(input_adv)
            loss = criterion(output, target)

        all_adv_image.append(input_adv.cpu().detach())
        all_target.append(target.cpu())

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print_and_log('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()


    all_adv_image = torch.cat(all_adv_image, dim=0)
    all_target = torch.cat(all_target, dim=0)
    print('Image shape = {}, Target shape = {}'.format(all_adv_image.shape, all_target.shape))

    print_and_log('Robust Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg, all_adv_image, all_target

def get_save_path(args):
    dir = ""
    if(args.fb):
        dir_format = 'fb_{args.arch}_{args.dataset}_d{args.density}_{args.growth}_T{args.update_frequency}_b{args.batch_size}_r{args.death_rate}_{flag}'
    elif(args.fbp):
        dir_format = 'fbp_{args.arch}_{args.dataset}_{args.sparse_init}_T{args.update_frequency}_d{args.density}_dr{args.death_rate}_{args.growth}_p{args.prune_ratio}_g{args.growth_ratio}_b{args.batch_size}_e{args.epoch_range}_r{args.update_threshold}_seed{args.seed}{flag}'
    else:
        dir_format = 'dense_{args.arch}_{args.dataset}_b{args.batch_size}_{flag}'

    dir = dir_format.format(args = args, flag = hashlib.md5(str(args).encode('utf-8')).hexdigest()[:4])
    save_path = os.path.join(args.save_dir, dir)
    return save_path

        
def input_a_sample(model, criterion, optimizer, args, data_sample):

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.eval()
    input, target = data_sample

    input = input.unsqueeze(dim = 0)
    target = torch.Tensor([target]).long()

    input = input.cuda()
    target = target.cuda()

    #adv samples
    with ctx_noparamgrad(model):
        input_adv = adversary.perturb(input, target)
    # compute output
    output_adv = model(input_adv)
    loss = criterion(output_adv, target)

    optimizer.zero_grad()
    loss.backward()

def get_generalization_gap(model, criterion, args):
   
    #final 
    train_loader, val_loader, test_loader, final_model = setup_dataset_models_eval(args)

    final_train_ra, _ = test_adv(train_loader, model, criterion, args)
    final_test_ra, _ = test_adv(test_loader, final_model, criterion, args)

    final_gap = final_train_ra - final_test_ra

    final_sparsity = check_sparsity(final_model)

    print('* Model final train RA = {:.2f}, final test RA = {:.2f}'.format(final_train_ra, final_test_ra))
    print('* Model final GAP = {:.2f}, final sparsity = {:.2f}'.format(final_gap, final_sparsity))

    return final_gap, final_sparsity


def getinfo(checkpoint):
    best_sa = checkpoint['best_sa']
    best_ra = checkpoint['best_ra']
    end_epoch = checkpoint['epoch']
    print('end_epoch', end_epoch)
    all_result = checkpoint['result']

    best_val_ra_index = all_result['val_ra'].index(best_ra)

    best_test_ra =  all_result['test_ra'][best_val_ra_index]
    final_test_ra = all_result['test_ra'][-1]
    diff1 = best_test_ra - final_test_ra


    best_test_sa =  all_result['test_sa'][best_val_ra_index]
    final_test_sa = all_result['test_sa'][-1]
    diff2 = best_test_sa - final_test_sa

    print('* Model best ra = {:.2f}, final_ra = {:.2f}, Diff1 = {:.2f}'.format(best_test_ra, final_test_ra, diff1))
    print('* Model best sa = {:.2f}, final_sa = {:.2f}, Diff2 = {:.2f}'.format(best_test_sa, final_test_sa, diff2))
