import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import random
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import json
import net.networks as networks
from utils.utils import *   
from net.csrnet import CSRNet
from utils.get_dataset import SHA_SSL 
import torch.nn.functional as F
from utils.generate_pseudo import generate_pseudo
from utils.generate_crop import generate_crop
import pickle
from train_val import validate

GPU = '0'
TEST_JSON = 'Dataset/SHA/SHA_train_unlabel_10%.json'
MODEL = 'result/SHA/10%/exp_seg_9/model_best_iteration_0.pth.tar'
TEST = False
parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--net_name', default='csrnet_uncer', type=str, help='path to latest checkpoint dir/files (default: none)')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = GPU

with open(TEST_JSON, 'r') as outfile:       
    test_idx_list = json.load(outfile)


transforms_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
test_dataset = SHA_SSL(train='val',
                        train_idx=test_idx_list,
                        transform_img=transforms_test)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                         sampler=torch.utils.data.SequentialSampler(test_dataset),
                                         batch_size=1,
                                         num_workers=8)
                   
# net = UncerCSRNet()
net = CSRNet()
net = net.cuda()
optimizer = torch.optim.Adam(net.parameters())
best_checkpoint = torch.load(MODEL)
net.load_state_dict(best_checkpoint['state_dict'])
optimizer.load_state_dict(best_checkpoint['optimizer'])
for param_group in optimizer.param_groups:
     cur_lr = param_group['lr']
print('begin test')
print('best epoch: {}'.format(best_checkpoint['epoch']))
print('best mae: {}'.format(best_checkpoint['best_mae']))
print('cur lr: {}'.format(cur_lr))
print('-'*40)

net.eval()

# mae, mse = validate(opt, test_loader, net)
# print('mae:', mae)
# print('mse:', mse)


pseudo_label_dict = generate_pseudo(opt, test_loader, net)
save_pseudo = 'result/SHA/10%/exp_seg_9/conf_2/'
os.makedirs(save_pseudo, exist_ok=True)
with open(os.path.join(save_pseudo, 'pseudo_labeling_iteration_1.pkl'), 'wb') as f:
    pickle.dump(pseudo_label_dict, f)


# save_crop = 'result/SHA/10%/exp_seg_9/conf_2/'
# os.makedirs(save_crop, exist_ok=True)
# crop_label_dict = generate_crop(save_crop+'crop1/', test_loader, net)
# with open(os.path.join(save_crop, 'crop_labeling_iteration_1.pkl'), 'wb') as f:
#     pickle.dump(crop_label_dict, f)
