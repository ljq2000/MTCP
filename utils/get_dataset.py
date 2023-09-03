import tarfile
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import h5py
import pickle
import os
import cv2
from utils.utils import *
import random


def get_sha(opt, label_idx_list, unlabel_idx_list, val_idx_list, pseudo_pkl, crop_pkl):

    transforms_img = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    label_idx_list = opt.xlabel*label_idx_list
    ori_pse_label_idx_len = len(label_idx_list)

    if pseudo_pkl is not None:

        pseudo_data_dict = pickle.load(open(pseudo_pkl, 'rb'))
        pseudo_label_idx = pseudo_data_dict['pseudo_imgpath_list']

        den_pseudo_idx = [None for i in range(len(label_idx_list))]
        den_pseudo_list = pseudo_data_dict['den_pseudo_list']
        den_pseudo_idx.extend(den_pseudo_list)

        seg_pseudo_idx = [None for i in range(len(label_idx_list))]
        seg_pseudo_list = pseudo_data_dict['seg_pseudo_list']
        seg_pseudo_idx.extend(seg_pseudo_list)

        confidence_idx = [None for i in range(len(label_idx_list))]
        confidence_list = pseudo_data_dict['confidence_list']
        confidence_idx.extend(confidence_list)

        ori_pse_label_idx_len = ori_pse_label_idx_len + len(pseudo_label_idx)

    else:
        pseudo_label_idx = None
        den_pseudo_idx = None
        confidence_idx= None
        seg_pseudo_idx=None
    

    if crop_pkl is not None:
        crop_data_dict = pickle.load(open(crop_pkl, 'rb'))
        crop_img_idx = crop_data_dict['crop_imgpath_list']
        crop_den_idx = [None for i in range(ori_pse_label_idx_len)]
        crop_density_list = crop_data_dict['crop_density_list']
        crop_den_idx.extend(crop_density_list)
    
    else:
        crop_img_idx = None
        crop_den_idx = None

    train_label_dataset = SHA_SSL(train='label',
                                  train_idx= label_idx_list,
                                  train_pseudo_idx=pseudo_label_idx,
                                  den_pseudo_idx=den_pseudo_idx,
                                  seg_pseudo_idx=seg_pseudo_idx,
                                  confidence_idx=confidence_idx,
                                  crop_img_idx=crop_img_idx,
                                  crop_den_idx=crop_den_idx,
                                  transform_img=transforms_img)

    unlabel_dataset = SHA_SSL(train='unlabel',
                                    train_idx=unlabel_idx_list,
                                    transform_img=transforms_img)

    val_dataset = SHA_SSL(train='val',
                          train_idx=val_idx_list,
                          transform_img=transforms_img)

    return train_label_dataset, unlabel_dataset, val_dataset

class SHA_SSL(Dataset):
    def __init__(self, 
                train='train', 
                train_idx=None,
                train_pseudo_idx=None,
                den_pseudo_idx=None,
                seg_pseudo_idx=None,
                confidence_idx=None,
                crop_img_idx=None,
                crop_den_idx=None,
                transform_img=None):

        self.train = train
        self.train_idx = train_idx
        self.train_pseudo_idx = train_pseudo_idx
        self.den_pseudo_idx = den_pseudo_idx
        self.seg_pseudo_idx = seg_pseudo_idx
        self.confidence_idx = confidence_idx
        self.crop_img_idx = crop_img_idx
        self.crop_den_idx = crop_den_idx
        
        if self.train_pseudo_idx is not None:
            self.train_idx.extend(self.train_pseudo_idx)
        if self.crop_img_idx is not None:
            self.train_idx.extend(self.crop_img_idx)


        self.transform_img = transform_img

    def __getitem__(self, idx):

        assert idx<=len(self.train_idx)
        img_path = self.train_idx[idx]
        img = Image.open(img_path).convert('RGB')

        if self.train == 'label':
            if (self.train_pseudo_idx is not None) and (img_path in self.train_pseudo_idx):
                flag = 'pse-label'
                density = np.asarray(self.den_pseudo_idx[idx])
                confidence = np.asarray(self.confidence_idx[idx])
                seg = np.asarray(self.seg_pseudo_idx[idx])
                confidence_fore_norm = normalize_numpy_nozero(confidence*((seg==1)*1))
                confidence_back_norm = normalize_numpy_nozero(confidence*((seg==0)*1))
                confidence = (confidence, confidence_fore_norm, confidence_back_norm)
                img = self.transform_img(img)
                return img, density, confidence, flag
            elif (self.crop_img_idx is not None) and (img_path in self.crop_img_idx):
                flag = 'crop-label'
                density = np.asarray(self.crop_den_idx[idx])
                density = cv2.resize(density,(density.shape[1]//8,density.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
                seg = (density>0)*1
                img = self.transform_img(img)
                return img, density, seg, flag
            else:
                flag = 'ori-label'
                img = self.transform_img(img)
                gt_file = h5py.File(img_path.replace('.jpg','.h5').replace('images','h5pys'), 'r')
                den_target = np.asarray(gt_file['density'])
                den_target = cv2.resize(den_target,(den_target.shape[1]//8,den_target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
                seg_target = (den_target>0)*1
                return img, den_target, seg_target, flag
        elif self.train == 'unlabel':
            img = self.transform_img(img)
            return img
        elif self.train == 'val':
            flag = 'val'
            img = self.transform_img(img)
            gt_file = h5py.File(img_path.replace('.jpg','.h5').replace('images','h5pys'), 'r')
            den_target = np.asarray(gt_file['density'])
            den_target = cv2.resize(den_target,(den_target.shape[1]//8,den_target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
            seg_target = (den_target>0)*1
            return img, den_target, seg_target, img_path, flag
        else:
            raise NotImplementedError('what ???')
        
    def __len__(self):
        return len(self.train_idx)

        
        


