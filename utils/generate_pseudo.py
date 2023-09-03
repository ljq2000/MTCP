import torch
import torch.nn.functional as F
from utils.utils import *
from tqdm import tqdm
import time
import cv2

def normalize(image, MIN_BOUND, MAX_BOUND):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # reverse_image = 1 - image
    # return reverse_image
    return image

def generate_pseudo(opt, data_loader, net):

    data_time = AverageMeter()
    end = time.time()

    pseudo_imgpath_list = []

    den_pseudo_list = []
    den_gt_list = []
    den_uncer_list = []

    seg_pseudo_list = []
    seg_uncer_list = []
    seg_gt_list = []

    confidence_list = []

    data_loader = tqdm(data_loader)
    net.eval()


    with torch.no_grad():
        for i,(img, den_gt, seg_gt, img_path, flag) in enumerate(data_loader):

            img = img.cuda()
            imgpath = img_path[0]

            pseudo_den = []
            for _ in range(5):
                img_den = img + torch.clamp(torch.randn_like(img) * 0.1, -0.2, 0.2)
                den_pred, _, _= net(img_den)
                pseudo_den.append(den_pred.squeeze(0).squeeze(0))
            den_uncer = torch.std(torch.stack(pseudo_den), dim=0)

            den_pred, seg_pred, uncer_pred = net(img)
            
            seg_pred = F.softmax(seg_pred, dim=1)
            pseudo_seg_val, pseudo_seg_idx = torch.max(seg_pred, dim=1) 
            seg_uncer = -1.0 * torch.sum(seg_pred * torch.log(seg_pred + 1e-6), dim=1)
            seg_uncer = normalize(seg_uncer, 0, np.log(2))
            confidence = torch.sigmoid(uncer_pred)


            den_gt = F.relu(den_gt.type(torch.FloatTensor))
            den_gt = den_gt.squeeze(0).data.cpu().numpy().tolist()
            den_pseudo = den_pred.squeeze(0).squeeze(0).data.cpu().numpy().tolist()
            den_uncer = den_uncer.data.cpu().numpy().tolist()
            seg_gt = seg_gt.squeeze(0).data.cpu().numpy().tolist()
            seg_pseudo = torch.where(pseudo_seg_idx==0, 0, 1).squeeze(0).data.cpu().numpy().tolist()
            seg_uncer = seg_uncer.squeeze(0).data.cpu().numpy().tolist()
            confidence = confidence.squeeze(0).squeeze(0).data.cpu().numpy().tolist()

        
            confidence_fore = np.asarray(confidence)*((np.asarray(seg_pseudo)==1)*1)
            confidence_fore = confidence_fore[np.nonzero(confidence_fore)]
            confidence_back = np.asarray(confidence)*((np.asarray(seg_pseudo)==0)*1)
            confidence_back = confidence_back[np.nonzero(confidence_back)]

            if len(confidence_fore) == 0 or len(confidence_back) == 0 :
                continue
    
            pseudo_imgpath_list.append(str(imgpath))
            den_pseudo_list.append(den_pseudo)
            den_gt_list.append(den_gt)
            den_uncer_list.append(den_uncer)
            seg_pseudo_list.append(seg_pseudo)
            seg_gt_list.append(seg_gt)
            seg_uncer_list.append(seg_uncer)
            confidence_list.append(confidence)

        # 存储
        pseudo_imgpath_list = np.array(pseudo_imgpath_list)
        den_pseudo_list = np.array(den_pseudo_list)
        den_gt_list = np.array(den_gt_list)
        den_uncer_list = np.array(den_uncer_list)
        seg_pseudo_list = np.array(seg_pseudo_list)
        seg_gt_list = np.array(seg_gt_list)
        seg_uncer_list = np.array(seg_uncer_list)
        confidence_list = np.array(confidence_list)
        

        pseudo_label_dict = {'pseudo_imgpath_list': pseudo_imgpath_list,
                            'den_pseudo_list': den_pseudo_list,
                            'den_gt_list': den_gt_list,
                            'den_uncer_list': den_uncer_list,
                            'seg_pseudo_list': seg_pseudo_list,
                            'seg_gt_list': seg_gt_list,
                            'seg_uncer_list': seg_uncer_list,
                            'confidence_list': confidence_list}
        
    return pseudo_label_dict


