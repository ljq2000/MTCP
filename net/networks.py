import torch
import torch.nn as nn
from net.csrnet import CSRNet as CSRNet


def select_optimizer(net, opt):

    """ select optimizer """

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, )
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(net.parameters(), opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError('This optimizer has not implemented yet')
    return optimizer


def define_net(opt):

    """select and define net"""

    if opt.net_name == 'csrnet':
            net = CSRNet()
    else:
        raise NotImplementedError('Unrecognized model: '+ opt.net_name)
    return net