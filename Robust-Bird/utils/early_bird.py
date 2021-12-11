import torch.nn as nn
import torch




# ######################################################################
# ============================= Early Bird =============================
# ######################################################################


class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model, percent):
        # prune based on BN layer
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        # save all BN weights
        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        # sort
        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index]

        # create mask
        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                index += size
        return mask

    def unstructure_pruning(self, model, percent):
        # count the total number of weights in Conv2D
        total = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                total += m.weight.data.numel()

        # save all weights
        conv_weights = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                size = m.weight.data.numel()
                conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
                index += size

        # sort
        y, i = torch.sort(conv_weights)
        thre_index = int(total * percent)
        thre = y[thre_index]

        # create mask
        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                index += size
        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        # the distance between two masks
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def structure_early_bird_emerge(self, model):
        # STEP1: get mask based on BN layer
        mask = self.pruning(model, self.percent)
        # STEP2: save current mask
        self.put(mask)
        #
        flag = self.cal_dist()
        if flag == True:
            print('EB mask distance ', self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False

    def unstructure_early_bird_emerge(self, model):
        # STEP1: get mask based on BN layer
        mask = self.unstructure_pruning(model, self.percent)
        # STEP2: save current mask
        self.put(mask)
        #
        flag = self.cal_dist()
        if flag == True:
            print('EB mask distance ', self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.08 :
                    return False
            return True
        else:
            return False

