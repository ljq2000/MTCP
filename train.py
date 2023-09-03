from math import log
import sys
import os
from urllib import parse
import warnings
import logging

import torch
# import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
import time
import datetime
import fitlog
from utils.get_dataset import SHA_SSL 
from utils.get_dataset import get_sha
import net.networks as networks
from train_val import *
from utils.utils import *
from net.csrnet import CSRNet
from utils.generate_pseudo import generate_pseudo
from utils.generate_crop import generate_crop
import pickle


parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--label_json', type=str, default='Dataset/SHA/SHA_train_label.json', help='path to train json')
parser.add_argument('--unlabel_json', type=str, default='Dataset/SHA/SHA_train_unlabel.json', help='path to train json')
parser.add_argument('--test_json', type=str,  default='Dataset/SHA/SHA_train_test.json', help='path to test json')
parser.add_argument('--crop_pkl', type=str,  default=None, help='path to test json')
parser.add_argument('--pre', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')
parser.add_argument('--start_epoch', type=int, default=0, help='Epochs of training start')
parser.add_argument('--epochs', type=int, default=200, help='Epochs of training end')
parser.add_argument('--print_per_step', type=int, default=1, help='print log after print_step')
parser.add_argument('--out_file', type=str, default='./result/', help='output log/model are saved here')
parser.add_argument('--data_name', type=str, default='SHA', help='datasets are trained here')
parser.add_argument('--iterations', default=10, type=int, help='number of total pseudo-labeling iterations to run')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--num_workers', type=int, default=8, help='threads for loading data')
parser.add_argument('--optimizer', type=str, default='adam', help='optimier [sgd|adam|adamW]')
parser.add_argument('--seed', type=int, default=-1, help="random seed (-1: don't use random seed)")

# lr
parser.add_argument('--lr', type=float, default=1e-6, help='initial learning rate for optimizer')
parser.add_argument('--lr_epoch', type=int, default=30, help='how much epoch update lr')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='value of lr update per lr_epoch')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay of adam')
parser.add_argument('--momentum', type=float, default=0.95, help='sgd momentum')


parser.add_argument('--xlabel', type=int, default=4, help='The label data multiplication --tln_mul during training ')
parser.add_argument('--per_label', default='50%', type=str, help='percent of label data, default 50%')
parser.add_argument('--task_id',type=str, default='0', help='task id to use. 0:label 1:label + unlabel')
parser.add_argument('--net_name', type=str, default='csrnet', help='It decides which models are used')
parser.add_argument('--alpha_ulb', default=0.1, type=float,  help='Proportion of segmentation loss')
parser.add_argument('--weight_con', default=3, type=float,  help='Proportion of segmentation loss')
parser.add_argument('--alpha_con', default=0.01, type=float,  help='Proportion of segmentation loss')
parser.add_argument('--alpha_seg', default=0.01, type=float,  help='Proportion of segmentation loss')
parser.add_argument('--epoch_con', default=100, type=int,  help='Proportion of segmentation loss')
parser.add_argument('--seg', default=False, type=bool,  help='Proportion of segmentation loss')

parser.add_argument('--resume', type=str, default=None, help='path to the pretrained model')



opt = parser.parse_args()

# 设置随机数
if opt.seed==-1:
    opt.seed = np.random.randint(1,211008)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
else:
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)


out_file = str(os.path.join(opt.out_file, opt.data_name, opt.per_label)) + '/'
exp_name = f'exp_{opt.task_id}'
out_file = os.path.join(out_file, exp_name) + '/'
os.makedirs(out_file, exist_ok=True)

start_itr = 0
if opt.resume and os.path.isdir(opt.resume):
    resume_files = os.listdir(opt.resume)
    resume_itrs = [int(item.replace('.pkl','').split('_')[-1]) for item in resume_files if 'pseudo_labeling_iteration' in item]
    if len(resume_itrs)>0:
        start_itr = max(resume_itrs)
    out_file = opt.resume


log_file = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file =os.path.join(out_file, log_file + '.log')
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

with open(log_file, 'a') as opt_file:
    opt_file.write(message)
    opt_file.write('\n')
set_logger(log_file)


fitlog_file = out_file + 'fitlog' + '/'
os.makedirs(fitlog_file, exist_ok=True)
fitlog.set_log_dir(fitlog_file)
fitlog.add_hyper(opt)
fitlog.add_hyper_in_file(__file__)



label_list = None
unlabel_list = None
val_list = None


with open(opt.label_json, 'r') as outfile:        
    label_list = json.load(outfile)
if opt.unlabel_json:
    with open(opt.unlabel_json, 'r') as outfile:        
        unlabel_list = json.load(outfile)
with open(opt.test_json, 'r') as outfile:       
    val_list = json.load(outfile)

# if opt.crop_pkl:
#     crop_label_data = str(opt.crop_pkl)
#     # start_itr = 1
# else:
#     crop_label_data = None

for itr in range(start_itr, opt.iterations):

    if itr>0:
        pseudo_label_data = f'{out_file}pseudo_labeling_iteration_{str(itr)}.pkl'
        crop_label_data = f'{out_file}crop_labeling_iteration_{str(itr)}.pkl'
        # crop_label_data = None

    else:
        pseudo_label_data = None
        crop_label_data = None



    train_label_dataset, unlabel_dataset, val_dataset = get_sha(opt,
                                                          label_idx_list=label_list,
                                                          unlabel_idx_list=unlabel_list,
                                                          val_idx_list=val_list,
                                                          pseudo_pkl=pseudo_label_data,
                                                          crop_pkl=crop_label_data)  

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    net = networks.define_net(opt)
    if len(opt.gpu)>1:
        str_ids = opt.gpu.split(',')
        gpu_list = []
        for str_id in str_ids:
            id = int(str_id)
            gpu_list.append(id)
        gpu_list = [i.item() for i in torch.arange(len(gpu_list))]
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda:0" if USE_CUDA else "cpu")
        net = torch.nn.DataParallel(net, device_ids=gpu_list)
        net.to(device)
    elif len(opt.gpu)==1:
        net = net.cuda()
    

    label_loader = torch.utils.data.DataLoader(train_label_dataset,
                                               sampler=torch.utils.data.RandomSampler(train_label_dataset),
                                               batch_size=opt.batch_size,
                                               num_workers=opt.num_workers)
    unlabel_loader = torch.utils.data.DataLoader(unlabel_dataset,
                                               sampler=torch.utils.data.SequentialSampler(unlabel_dataset),
                                               batch_size=opt.batch_size,
                                               num_workers=opt.num_workers)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             sampler=torch.utils.data.SequentialSampler(val_dataset),
                                             batch_size=opt.batch_size,
                                             num_workers=opt.num_workers)


    optimizer = networks.select_optimizer(net, opt)
    if opt.pre:
        if os.path.isfile(opt.pre):
            print("=> loading checkpoint '{}'".format(opt.pre))
            checkpoint = torch.load(opt.pre)
            opt.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_mae']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print( f'=> loaded checkpoint {opt.pre} epoch {opt.start_epoch}')
        else:
            print(f'=> no checkpoint found at {opt.pre}')

    best_mae = 1e6
    best_mse = 1e6
    net.zero_grad()
    for epoch in range(opt.start_epoch, opt.epochs):

        logging.info('-'*5 + f'Inter {itr}/{opt.iterations}  Epoch {epoch}/{opt.epochs-1}' + '-'*5)
        adjust_learning_rate(optimizer, epoch, opt.lr, opt.lr_epoch, opt.lr_decay_rate)        
                
        if itr==0:
            loss, loss_lb, lossden_lb, lossseg_lb, lossescon_lb = train(opt, label_loader, net, optimizer, epoch, itr)
            fitlog.add_loss({'train loss': loss}, step=(itr*opt.epochs)+epoch)
            fitlog.add_loss({'train loss_lb': loss_lb}, step=(itr*opt.epochs)+epoch)
            fitlog.add_loss({'train lossden_lb': lossden_lb}, step=(itr*opt.epochs)+epoch)
            fitlog.add_loss({'train lossseg_lb': lossseg_lb}, step=(itr*opt.epochs)+epoch)
            fitlog.add_loss({'train lossescon_lb': lossescon_lb}, step=(itr*opt.epochs)+epoch)
        else:
            loss, loss_lb, loss_ulb = train_ssl(opt, label_loader, net, optimizer, epoch, itr)
            fitlog.add_loss({'train loss': loss}, step=(itr*opt.epochs)+epoch)
            fitlog.add_loss({'train loss_lb': loss_lb}, step=(itr*opt.epochs)+epoch)
            fitlog.add_loss({'train loss_ulb': loss_ulb}, step=(itr*opt.epochs)+epoch)

        if epoch>-1:
            mae, mse = validate(opt, val_loader, net)
            fitlog.add_metric({'val mae': mae}, step=(itr*opt.epochs)+epoch)
            fitlog.add_metric({'val mse': mse}, step=(itr*opt.epochs)+epoch)
            is_best = mae < best_mae
            best_mae = min(mae, best_mae)
            if is_best:
                best_mse = mse
                fitlog.add_best_metric({'val best mae': best_mae})
            logging.info(' * best MAE {mae:.3f} '.format(mae=best_mae))
            logging.info(' * best MSE {mse:.3f} '.format(mse=best_mse))
            logging.info(' * Loss All {loss:.3f} '.format(loss=loss))
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': net.state_dict(),
                            'best_mae': best_mae,
                            'optimizer' : optimizer.state_dict()}, 
                            # 'scheduler' : scheduler.state_dict()}, 
                            is_best,
                            out_file,
                            itr,
                            f'checkpoint_iteration_{str(itr)}.pth.tar')
        fitlog.finish()

    best_checkpoint = torch.load(f'{out_file}model_best_iteration_{str(itr)}.pth.tar')
    net.load_state_dict(best_checkpoint['state_dict'])
    net.eval()

    transforms_img = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    with open(opt.label_json, 'r') as outfile:       
        label_idx_list = json.load(outfile)
    with open(opt.unlabel_json, 'r') as outfile:       
        unlabel_idx_list = json.load(outfile)
    
    label_dataset = SHA_SSL(train='val',
                    train_idx=label_idx_list,
                    transform_img=transforms_img)
    unlabel_dataset = SHA_SSL(train='val',
                    train_idx=unlabel_idx_list,
                    transform_img=transforms_img)

    label_loader = torch.utils.data.DataLoader(label_dataset,
                                        sampler=torch.utils.data.SequentialSampler(label_dataset),
                                        batch_size=1,
                                        num_workers=8)
    unlabel_loader = torch.utils.data.DataLoader(unlabel_dataset,
                                        sampler=torch.utils.data.SequentialSampler(unlabel_dataset),
                                        batch_size=1,
                                        num_workers=8)
    save_path = opt.resume
    os.makedirs(save_path, exist_ok=True)
    crop_label_dict = generate_crop(save_path+f'crop{itr+1}/', label_loader, net)
    with open(os.path.join(save_path, f'crop_labeling_iteration_{itr+1}.pkl'), 'wb') as f:
        pickle.dump(crop_label_dict, f)


    pseudo_label_dict = generate_pseudo(opt, unlabel_loader, net)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f'pseudo_labeling_iteration_{itr+1}.pkl'), 'wb') as f:
        pickle.dump(pseudo_label_dict, f)
    
    logging.info(f'############################# PL Iteration: {itr+1} #############################\n')









            




