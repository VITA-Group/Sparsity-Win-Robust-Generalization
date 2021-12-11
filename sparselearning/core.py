from __future__ import print_function
from numpy.core.records import array
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import math
from sparselearning.igq import get_igq_sparsities
import pdb


class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, death_rate):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate



class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', args=None, spe_initial=None, train_loader=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.loader = train_loader
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        self.spe_initial = spe_initial # initial masks made by SNIP
        self.snip_masks = None # masks made by SNIP during training

        self.masks = {}
        self.newly_masks = {}
        self.survival = {}
        self.pruned_number = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        # stats
        self.name2zeros = {}
        self.name2nonzeros = {}

        self.death_rate = death_rate
        self.name2death_rate = {}
        self.steps = 0
        #dst sparse
        self.start_epoch_dst = 0
        self.current_epoch = 0
        self.prune_ratio = self.args.prune_ratio
        self.growth_ratio = self.args.growth_ratio
        self.lossinfo = []
        self.train_val_diff = []
        self.update_threshold = self.args.update_threshold
        self.prune_epochs = []
        self.growth_epochs = []
        self.total_fired_weights = 0
        self.init_mask = None

    def init(self, mode='ER', density=0.05, erk_power_scale=1.0):
        self.sparsity = density
        if mode == 'uniform':
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    self.masks[name_cur][:] = (torch.rand(weight.shape) < density).float().data.cuda()

        elif mode == 'fixed_ERK':
            print('initialize by fixed_ERK')
            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
            is_epsilon_valid = False
            dense_layers = set()
            while not is_epsilon_valid:
                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        elif mode == 'ER':
            print('initialize by SET')
            # initialization used in sparse evolutionary training
            total_params = 0
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    total_params += weight.numel()

            target_params = total_params *density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not ((current_params+tolerance > target_params) and (current_params-tolerance < target_params)):
                new_nonzeros = 0.0
                index = 0
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth =  epsilon*sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95
                

            index = 0
            for name, weight in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                growth =  epsilon*sum(weight.shape)
                prob = growth/np.prod(weight.shape)
                self.masks[name_cur][:] = (torch.rand(weight.shape) < prob).float().data.cuda()
                
        # random igq
        elif mode == 'igq':
            model = self.modules[0]
            sparsities = get_igq_sparsities(model, density)
            print(sparsities)
            it = iter(sparsities)
            index = 0
            for name, weight in model.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                mask = torch.ones(weight.shape)

                ind = np.random.choice(range(np.prod(mask.shape)), 
                    size=int(next(it)*np.prod(mask.shape)), replace=False)
                mask.reshape(-1)[ind] = 0.
                self.masks[name_cur][:] = mask.float().data.cuda()

        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks) # used for over-paremeters
        self.init_death_rate(self.death_rate)

        self.gather_statistics()
        self.print_nonzero_counts()

        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1}'.format(density, sparse_size / total_size))

    def get_info_resume(self):
        info = {
            'lossinfo': self.lossinfo,
            'train_val_diff': self.train_val_diff,
            'prune_epochs': self.prune_epochs,
            'growth_epochs': self.growth_epochs,
            'masks': self.masks,
            'name2death_rate': self.name2death_rate,
            'decay': (self.death_rate_decay.sgd.state_dict(), self.death_rate_decay.cosine_stepper.state_dict()),
        }

        return info

    def load_info_resume(self, info):
        self.lossinfo = info['lossinfo']
        self.train_val_diff = info['train_val_diff']
        self.prune_epochs = info['prune_epochs']
        self.growth_epochs = info['growth_epochs']
        self.masks = info['masks']
        self.name2death_rate = info['name2death_rate']
        self.death_rate_decay.sgd.load_state_dict(info['decay'][0])
        self.death_rate_decay.cosine_stepper.load_state_dict(info['decay'][1])


    def init_death_rate(self, death_rate):
        for name in self.masks:
            self.name2death_rate[name] = death_rate

    def at_end_of_epoch(self):
        self.truncate_weights()
        _, total_fired_weights = self.fired_masks_update()
        self.total_fired_weights = total_fired_weights
        self.print_nonzero_counts()


    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        for name in self.masks:
            if self.args.decay_schedule == 'cosine':
                self.name2death_rate[name] = self.death_rate_decay.get_dr(self.name2death_rate[name])
            elif self.args.decay_schedule == 'constant':
                self.name2death_rate[name] = self.args.death_rate
            self.death_rate = self.name2death_rate[name]
        self.steps += 1

    def add_module(self, module, density, sparse_init='ER'):
        self.modules.append(module)
        index = 0
        for name, tensor in module.named_parameters():
            name_cur = name + '_' + str(index)
            index += 1
            if len(tensor.size()) ==4 or len(tensor.size()) ==2:
                self.names.append(name_cur)
                self.masks[name_cur] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        self.init(mode=sparse_init, density=density)

    def remove_weight(self, name, index):

        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        index = 0
        for module in self.modules:
            for name, module in module.named_modules():
                print(name)
                if isinstance(module, nn_type):
                    self.remove_weight(name, index)
                index += 1

    def apply_mask(self):
        index = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                name_cur = name+'_'+str(index)
                index += 1
                if name_cur in self.masks:
                    weight.data = weight.data*self.masks[name_cur]
                    if 'momentum_buffer' in self.optimizer.state[weight]:
                        self.optimizer.state[weight]['momentum_buffer'] = self.optimizer.state[weight]['momentum_buffer']*self.masks[name_cur]

##################### Flying Bird+ sparse ################
    def set_dst_start_epoch(self, epoch):
        self.start_epoch_dst = epoch
    
    def set_dst_current_epoch(self, epoch):
        self.current_epoch = epoch

    def update_loss_info(self, loss):
        self.lossinfo.append(loss)
    
    def update_train_val_diff(self, train_ra, val_ra):
        self.train_val_diff.append(train_ra - val_ra)
        # self.train_val_diff.append(val_ra)
    
    def clear_dst_info(self):
        l0 = self.lossinfo[-1]
        l1 = self.train_val_diff[-1]
        self.lossinfo.clear()
        self.train_val_diff.clear()

        self.lossinfo.append(l0)
        self.train_val_diff.append(l1)

    def get_dst_ratio(self):
        r = self.args.epoch_range + 1
        # growth_ratio
        l0 = self.lossinfo[-r:][:-1]
        l1 = self.lossinfo[-r:][1:]

        diff1 = np.array(l1) - np.array(l0)
        c = np.sum(diff1 >= 0)
        growth_ratio = c / float(diff1.size)

        # prune_ratio
        l0 = self.train_val_diff[-r:][:-1]
        l1 = self.train_val_diff[-r:][1:]

        diff1 = np.array(l1) - np.array(l0)
        # c = np.sum(diff1 < 0)
        c = np.sum(diff1 > 0)

        prune_ratio = c / float(diff1.size)
        

        return prune_ratio, growth_ratio


    def dst_prune(self):
        if not self.args.dynamic_sparse or self.current_epoch < self.start_epoch_dst:
            return False
        
        prune_ratio, _ = self.get_dst_ratio()

        return self.update_threshold <= prune_ratio
    
    def dst_growth(self):
        if not self.args.dynamic_sparse or self.current_epoch < self.start_epoch_dst:
            return False
        
        _, growth_ratio = self.get_dst_ratio()
        
        return self.update_threshold <= growth_ratio
        # return False

    def get_dst_epochs(self):
        return self.prune_epochs, self.growth_epochs
        

#####################################################################

    def truncate_weights(self):
        self.gather_statistics()
        #prune
        index = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                mask = self.masks[name_cur]

                # death
                if self.death_mode == 'magnitude':
                    new_mask, dst_new_mask = self.magnitude_death(mask, weight, name_cur)

                self.pruned_number[name_cur] = int(self.name2nonzeros[name_cur] - new_mask.sum().item())
                if self.dst_prune():
                    self.prune_epochs.append(self.current_epoch)
                    self.masks[name_cur][:] = dst_new_mask
                else:
                    self.masks[name_cur][:] = new_mask
        

        #grow
        index = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                name_cur = name +'_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                new_mask = self.masks[name_cur].data.byte()


                if self.growth_mode == 'random':
                    new_mask, dst_new_mask = self.random_growth(name_cur, new_mask, self.pruned_number[name_cur], weight)

                elif self.growth_mode == 'gradient':
                    # implementation for Rigging Ticket
                    new_mask, dst_new_mask = self.gradient_growth(name_cur, new_mask, self.pruned_number[name_cur], weight)

                # exchanging masks
                self.masks.pop(name_cur)
                if self.args.dynamic_sparse and self.dst_growth():
                    self.growth_epochs.append(self.current_epoch)
                    self.masks[name_cur] = dst_new_mask.float()
                else:
                    self.masks[name_cur] = new_mask.float()

        if self.args.dynamic_sparse:
            if self.dst_prune():
                print("epoch{} dst prune!".format(self.current_epoch))
            if self.dst_growth():
                print("epoch{} dst growth!".format(self.current_epoch))
            
            # self.clear_dst_info()

        self.apply_mask()


    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}

        index = 0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                mask = self.masks[name_cur]
                self.name2nonzeros[name_cur] = mask.sum().item()
                self.name2zeros[name_cur] = mask.numel() - self.name2nonzeros[name_cur]



    '''
                    DEATH
    '''
    def CS_death(self,  mask,  snip_mask):
        # calculate scores for all weights
        # note that the gradients are from the last iteration, which are not very accurate
        # but in another perspective, we can understand the weights are from the next iterations, the differences are not very large.
        '''
        grad = self.get_gradient_for_weights(weight)
        scores = torch.abs(grad * weight * (mask == 0).float())
        norm_factor = torch.sum(scores)
        scores.div_(norm_factor)
        x, idx = torch.sort(scores.data.view(-1))

        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        if num_remove == 0.0: return weight.data != 0.0

        mask.data.view(-1)[idx[:k]] = 0.0
        '''

        assert (snip_mask.shape == mask.shape)

        return snip_mask

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def magnitude_death(self, mask, weight, name):

        if mask.sum().item() == mask.numel():
            return mask, mask

        death_rate = self.name2death_rate[name]

        num_remove = math.ceil(death_rate*self.name2nonzeros[name])
        dst_remove = math.ceil(death_rate*self.prune_ratio*self.name2nonzeros[name])
        # pdb.set_trace()
        if num_remove == 0.0: return weight.data != 0.0, weight.data != 0.0
        #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        dst_k = min(math.ceil(num_zeros + num_remove + dst_remove), n)

        threshold = x[k-1].item()
        dst_threshold = x[dst_k-1].item()
        # print("should prune:", num_remove)
        return (torch.abs(weight.data) > threshold), (torch.abs(weight.data) > dst_threshold)
    
    def part_magnitude_death(self, mask, weight, name):

        if mask.sum().item() == mask.numel():
            return mask, mask

        death_rate = self.name2death_rate[name]

        num_remove = math.ceil(death_rate*self.name2nonzeros[name]) # pruning nonzeros
        dst_remove = math.ceil(death_rate*self.prune_ratio*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0, weight.data != 0.0
        #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])

        rand_num_remove = int(0.1*num_remove)
        mag_num_remove = num_remove - rand_num_remove
        num_zeros = self.name2zeros[name]

        # rand-prune
        rand_weight = torch.rand(weight.shape).cuda()*mask
        rand_x, rand_idx = torch.sort(torch.abs(rand_weight.view(-1)))
        rand_n = rand_idx.shape[0]
        rand_k = math.ceil(rand_num_remove + num_zeros)
        rand_threshold = rand_x[rand_k-1].item()
        new_mask = (torch.abs(rand_weight) > rand_threshold).float()

        # magnitude_prune
        new_weight = weight*new_mask
        x, idx = torch.sort(torch.abs(new_weight.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        dst_K =  math.ceil(num_zeros + num_remove + dst_remove)
        threshold = x[k-1].item()
        dst_threshold = x[dst_K-1].item()

        return (torch.abs(new_weight) > threshold), (torch.abs(new_weight) > dst_threshold)

    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_growth(self, name, new_mask, total_regrowth, weight):
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask, new_mask
        dst_total_regrowth = min(total_regrowth*(1.0+self.growth_ratio), n)
        expeced_growth_probability = (total_regrowth/n)
        dst_expeced_growth_probability = (dst_total_regrowth/n)

        r = torch.rand(new_mask.shape).cuda()
        new_weights = r < expeced_growth_probability
        dst_new_weights = r < dst_expeced_growth_probability

        # for pytorch1.5.1, use return new_mask.bool() | new_weights
        result_mask = new_mask.byte() | new_weights
        dst_result_mask = new_mask.byte() | dst_new_weights
        # print("total_regrowth: {0}, real_growth: {1}".format(total_regrowth, result_mask.sum() - new_mask.sum()))
        return result_mask, dst_result_mask

    def momentum_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        if total_regrowth == 0:
            return new_mask, new_mask
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        n = idx.shape[0]
        dst_total_regrowth = min(int(total_regrowth*(1.0+self.growth_ratio)), n)

        result_mask = copy.deepcopy(new_mask)
        # print('threshold value : ', result_mask.data.view(-1)[idx[total_regrowth]])
        # print('the next value is :', result_mask.data.view(-1)[idx[total_regrowth + 1]])
        
        result_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        # act_growth = result_mask.sum().item() - new_mask.sum().item()

        # print('act growth: ', act_growth)
      

        dst_result_mask = copy.deepcopy(new_mask)
        dst_result_mask.data.view(-1)[idx[:dst_total_regrowth]] = 1.0

        return result_mask, dst_result_mask
    
    def part_gradient_growth(self, name, new_mask, total_regrowth, weight):
        if total_regrowth == 0:
            return new_mask, new_mask

        grad = self.get_gradient_for_weights(weight)

        rand_regrowth = int(0.1*total_regrowth)
        grad_regrowth = total_regrowth - rand_regrowth
        rand_grad = torch.rand_like(grad)
        rand_grad = rand_grad*(new_mask == 0).float()
        rand_y, rand_idx = torch.sort(torch.abs(rand_grad).flatten(), descending=True)
        new_mask.data.view(-1)[rand_idx[:rand_regrowth]] = 1.0

        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        n = idx.shape[0]
        dst_total_regrowth = min(int(total_regrowth*(1.0+self.growth_ratio)), n)
        dst_grad_regrowth = dst_total_regrowth - rand_regrowth

        result_mask = copy.deepcopy(new_mask)
        result_mask.data.view(-1)[idx[:grad_regrowth]] = 1.0

        dst_result_mask = copy.deepcopy(new_mask)
        dst_result_mask.data.view(-1)[idx[:dst_grad_regrowth]] = 1.0

        return result_mask, dst_result_mask

    def mix_growth(self, name, new_mask, total_regrowth, weight):
        gradient_grow = int(total_regrowth * self.args.mix)
        random_grow = total_regrowth - gradient_grow
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (new_mask == 0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:gradient_grow]] = 1.0

        n = (new_mask == 0).sum().item()
        expeced_growth_probability = (random_grow / n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask = new_mask.byte() | new_weights

        return new_mask, grad

    def momentum_neuron_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        index = 0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                mask = self.masks[name_cur]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name_cur, self.name2nonzeros[name_cur], num_nonzeros, num_nonzeros/float(mask.numel()))
                print(val)
        print('Death rate: {0}\n'.format(self.death_rate))


    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        index = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                self.fired_masks[name_cur] = self.masks[name_cur].data.byte() | self.fired_masks[name_cur].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name_cur].sum().item())
                ntotal_weights += float(self.fired_masks[name_cur].numel())
                layer_fired_weights[name_cur] = float(self.fired_masks[name_cur].sum().item())/float(self.fired_masks[name_cur].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name_cur])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights