import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image  
import numpy as np
import cv2
from utils.utils import *
import time
from tqdm import tqdm

def generate_crop(out_file, data_loader, net):
    
    pseudo_imgpath_list = []
    confidence_list = []

    data_loader = tqdm(data_loader)
    net.eval()

    with torch.no_grad():
        for i,(img, den_gt, seg_gt, img_path, flag) in enumerate(data_loader):

            img = img.cuda()
            imgpath = img_path[0]

            _, _, uncer_pred = net(img)
            confidence = torch.sigmoid(uncer_pred)
            confidence = confidence.squeeze(0).squeeze(0).data.cpu().numpy().tolist()

            pseudo_imgpath_list.append(str(imgpath))
            confidence_list.append(confidence)

        pseudo_imgpath_list = np.array(pseudo_imgpath_list)
        confidence_list = np.array(confidence_list)

    out_file_images = out_file + 'images/'
    out_file_h5pys = out_file + 'h5pys/'
    out_file_show = out_file + 'show/'
    os.makedirs(out_file, exist_ok=True)
    os.makedirs(out_file_images, exist_ok=True)
    os.makedirs(out_file_h5pys, exist_ok=True)
    os.makedirs(out_file_show, exist_ok=True)
    crop_label_dict = crop_dict(out_file, pseudo_imgpath_list, confidence_list)

    return crop_label_dict


def crop_dict(outfile, pseudo_imgpath_list, confidence_list):
    
    crop_imgpath_list = []
    crop_density_list = []
    for idx in range(len(pseudo_imgpath_list)):
        imgpath, pred_confidence = pseudo_imgpath_list[idx], confidence_list[idx]

        img = cv2.imread(imgpath)
        pred_confidence = np.asarray(pred_confidence)
        confidence = 255*pred_confidence[10:-10, 10:-10]
        confidence_img  = confidence.astype(np.uint8)
        
        confidence_gray = confidence_img
        thresh = 0.6*255 
        ret, binary = cv2.threshold(confidence_gray, thresh, 255, cv2.THRESH_BINARY_INV)
        # cv2.imwrite('binary.png', binary)
        contours, hierarchy= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        if not len(contours):
            continue

        list_idx = []
        for i in range(len(contours)):
            list_idx.append(areaCal(contours[i]))
        list_idx.sort(reverse = True)
        # print(list_idx)

        for i in range(len(contours)):
            if areaCal(contours[i])==list_idx[0]:
                first_idx = i

        confidence_rgb = cv2.cvtColor(confidence_img, cv2.COLOR_BGR2RGB)
        # cv2.drawContours(confidence_rgb, contours[-1], -1, (0, 0, 255), 3)
        x, y, w, h = cv2.boundingRect(contours[first_idx])
        cv2.rectangle(confidence_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)

        xc, yc, wc, hc = (x+10)*8, (y+10)*8, w*8, h*8
        img_corp = img[yc:yc+hc, xc:xc+wc]
        # cv2.rectangle(img, (xc, yc), (xc+wc, yc+hc), (0, 0, 255), 2)

        gt_file = h5py.File(imgpath.replace('.jpg','.h5').replace('images','h5pys'), 'r')
        den_target = np.asarray(gt_file['density'])
        den_crop = den_target[yc:yc+hc, xc:xc+wc]

        img_name = imgpath.split('/')[-1]
        save_path = os.path.join(outfile, 'images', img_name)
        cv2.imwrite(save_path, img_corp)

        # with h5py.File(save_path.replace('.jpg','.h5').replace('images','h5pys'), 'w') as hf:
        #         hf['density'] = den_crop

        fore_idx = np.argwhere(den_crop>0)
        for coor in fore_idx:
            cv2.circle(img_corp, (int(coor[1]), int(coor[0])), 1, (0,255,0), 1)
        cv2.imwrite(save_path.replace('images', 'show'), img_corp)

        img_path = os.path.join('/root/CUPCC/', save_path)
        print(img_path)

        crop_imgpath_list.append(img_path)
        crop_density_list.append(den_crop)
    
    crop_imgpath_list = np.array(crop_imgpath_list)
    crop_density_list = np.array(crop_density_list)
    crop_label_dict = {'crop_imgpath_list': crop_imgpath_list,
                        'crop_density_list': crop_density_list}
    return crop_label_dict
        



