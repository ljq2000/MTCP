import h5py
import torch
import shutil
import os
import argparse
import logging
import math
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import cv2


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
  
def save_checkpoint(state, is_best, out_file, itr, filename='checkpoint.pth.tar'):
    torch.save(state, out_file+filename)
    if is_best:
        shutil.copyfile(out_file+filename, out_file+f'model_best_iteration_{str(itr)}.pth.tar')        

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def set_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s",
                                     "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)


def print_options(parser, opt, log_dir, exp_dir):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        with open(log_dir, 'a') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

def adjust_learning_rate(optimizer, epoch, lr_ori, lr_epoch=30, lr_decay_rate=0.5):
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    lr_cur = lr_ori * (lr_decay_rate ** (epoch // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_cur


# def den_to_seg(density_map, threshold):

#     density_mask = (density_map != threshold)
#     return density_mask

def den_to_seg(density_map, threshold):
    
    # density_mask = (density_map != threshold).long()
    density_mask = (density_map > threshold).long()
    density_mask = density_mask.squeeze(0)
    return density_mask

def den_to_back(density_map, threshold):
    
    # density_mask = (density_map != threshold).long()
    density_mask = (density_map == threshold).long()
    density_mask = density_mask.squeeze(0)
    return density_mask


def gumbel_softmax(logits, tau=1, hard=False, dim=1, training=True):
    
    """ See `torch.nn.functional.gumbel_softmax()` """
    # if training:
    # gumbels = -torch.empty_like(logits,
    #                             memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    # gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # # else:
    # #     gumbels = logits
    # y_soft = gumbels.softmax(dim)

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    with torch.no_grad():
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        #  **test**
        # index = 0
        # y_hard = torch.Tensor([1, 0, 0, 0]).repeat(logits.shape[0], 1).cuda()
    ret = y_hard - y_soft.detach() + y_soft  # 让其可导
    return y_soft, ret, index


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

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


def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index,layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x

def normalize_numpy_nozero(numpy_array):

    numpy_max = np.max(numpy_array.max())
    numpy_min = np.min(numpy_array[np.nonzero(numpy_array)])
    numpy_norm = (numpy_array -numpy_min)/(numpy_max-numpy_min)
    numpy_norm = 1*(numpy_norm>0)*numpy_norm
    return numpy_norm 

def areaCal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour)
    return area