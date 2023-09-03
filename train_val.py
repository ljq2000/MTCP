import logging

import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from utils.utils import *
from utils.losses import *
from utils.metrics import SegmentationMetric


def train(opt, train_loader, net, optimizer, epoch, itr):
    
    losses = AverageMeter()
    losses_lb = AverageMeter()
    lossesden_lb = AverageMeter()
    lossesseg_lb = AverageMeter()
    lossescon_lb = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
     
    
    criterion_den = torch.nn.MSELoss(reduction='sum').cuda()
    criterion_seg = torch.nn.CrossEntropyLoss(reduction='sum',ignore_index=10).cuda()
    criterion_con = SelfConfidMSELoss()
    criterion_nll =  torch.nn.MSELoss(reduction='none').cuda()

    for param_group in optimizer.param_groups:
        lr_cur = param_group['lr']

    print('itr %d epoch %d, processed %d samples, lr %.10f' % (itr, epoch, epoch * len(train_loader), lr_cur))

    net.train()

    for i,(img, den_gt, seg_gt, flag) in enumerate(train_loader):
        data_time.update(time.time() - end)
        flag = flag[0]
        img = img.cuda()

        if flag == 'ori-label':
            den_gt = den_gt.type(torch.FloatTensor).unsqueeze(0).cuda()
            den_gt = F.relu(den_gt)
            seg_gt = seg_gt.cuda()
            den_pred, seg_pred, con_pred = net(img)
            if opt.seg:
                lossden_lb = criterion_den(den_pred, den_gt)
                lossseg_lb = criterion_seg(seg_pred, seg_gt)
                if epoch>opt.epoch_con:
                    losscon_lb = criterion_con(seg_pred, con_pred, seg_gt)
                    loss_lb = 1*lossden_lb + opt.alpha_seg*lossseg_lb  + opt.alpha_con*losscon_lb
                    lossescon_lb.update(losscon_lb.item())
                else:
                    loss_lb = 1*lossden_lb + opt.alpha_seg*lossseg_lb
                lossesden_lb.update(lossden_lb.item())
                lossesseg_lb.update(lossseg_lb.item())
            else:
                lossden_lb = criterion_den(den_pred, den_gt)
                lossesden_lb.update(lossden_lb.item())
                loss_lb = 1*lossden_lb
            losses_lb.update(loss_lb.item())

        loss = loss_lb
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_per_step == 0:
            print('Itr: [{0}/{1}] \t'
                  'Epoch: [{2}][{3}]   [{4}/{5}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(itr, opt.iterations,
                          epoch, opt.epochs, i, len(train_loader), 
                          batch_time=batch_time,
                          data_time=data_time, 
                          loss=losses))

    return losses.avg, losses_lb.avg, lossesden_lb.avg, lossesseg_lb.avg, lossescon_lb.avg


def train_ssl(opt, train_loader, net, optimizer, epoch, itr):
    
    losses = AverageMeter()
    losses_lb = AverageMeter()
    losses_ulb = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
     
    
    criterion_den = torch.nn.MSELoss(reduction='sum').cuda()
    criterion_seg = torch.nn.CrossEntropyLoss(reduction='sum',ignore_index=10).cuda()
    criterion_con = SelfConfidMSELoss()
    criterion_den_ulb =  torch.nn.MSELoss(reduction='none').cuda()

    for param_group in optimizer.param_groups:
        lr_cur = param_group['lr']

    print('itr %d epoch %d, processed %d samples, lr %.10f' % (itr, epoch, epoch * len(train_loader), lr_cur))

    net.train()

    for i,(img, den_gt, seg_gt, flag) in enumerate(train_loader):
        data_time.update(time.time() - end)
        flag = flag[0]
        img = img.cuda()

        # 处理原始有标注数据
        if flag == 'ori-label':
            den_gt = den_gt.type(torch.FloatTensor).unsqueeze(0).cuda()
            den_gt = F.relu(den_gt)
            seg_gt = seg_gt.cuda()
            den_pred, seg_pred, con_pred = net(img)
            lossden_lb = criterion_den(den_pred, den_gt)
            lossseg_lb = criterion_seg(seg_pred, seg_gt)
            if epoch>opt.epoch_con:
                losscon_lb = criterion_con(seg_pred, con_pred, seg_gt)
                loss_lb = 1*lossden_lb + opt.alpha_seg*lossseg_lb  + opt.alpha_con*losscon_lb
            else:
                loss_lb = 1*lossden_lb + opt.alpha_seg*lossseg_lb
            losses_lb.update(loss_lb.item())

            loss = loss_lb
            losses.update(loss.item())

        elif flag == 'pse-label':
            den_pse = den_gt.type(torch.FloatTensor).unsqueeze(0).cuda()
            confidence, confidence_fore_norm, confidence_back_norm = seg_gt
            confidence_fore_norm = opt.weight_con*confidence_fore_norm.type(torch.FloatTensor).unsqueeze(0).cuda()
            confidence_back_norm = opt.weight_con*confidence_back_norm.type(torch.FloatTensor).unsqueeze(0).cuda()

            den_pred, _, _ = net(img)
            lossden_ulb = (confidence_fore_norm*criterion_den_ulb(den_pred, den_pse) + confidence_back_norm*criterion_den_ulb(den_pred, den_pse)).sum()
            loss_ulb = opt.alpha_ulb*lossden_ulb

            losses_ulb.update(loss_ulb.item())
            loss = loss_ulb
            losses.update(loss.item())

        elif flag == 'crop-label':
            den_gt = den_gt.type(torch.FloatTensor).unsqueeze(0).cuda()
            den_gt = F.relu(den_gt)
            seg_gt = seg_gt.cuda()
            den_pred, seg_pred, con_pred = net(img)
            lossden_lb = criterion_den(den_pred, den_gt)
            lossseg_lb = criterion_seg(seg_pred, seg_gt)
            if epoch>opt.epoch_con:
                losscon_lb = criterion_con(seg_pred, con_pred, seg_gt)
                loss_lb = 1*lossden_lb + opt.alpha_seg*lossseg_lb  + opt.alpha_con*losscon_lb
            else:
                loss_lb = 1*lossden_lb + opt.alpha_seg*lossseg_lb

            loss = loss_lb
            losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_per_step == 0:
            print('Itr: [{0}/{1}] \t'
                  'Epoch: [{2}][{3}]   [{4}/{5}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(itr, opt.iterations,
                          epoch, opt.epochs, i, len(train_loader), 
                          batch_time=batch_time,
                          data_time=data_time, 
                          loss=losses))

    return losses.avg, losses_lb.avg, losses_ulb.avg,


def validate(opt, val_loader, net):
    print ('begin test')
    net.eval()
    mae = 0
    mse = 0
    for i,(img, den_gt, _, _, _) in enumerate(val_loader):
        img = img.cuda()
        den_gt = den_gt.type(torch.FloatTensor).unsqueeze(0).cuda()
        den_gt = F.relu(den_gt)

        den_pred, _, _,= net(img)
        den_pred = den_pred.data.cpu().numpy()
        den_gt = den_gt.data.cpu().numpy()
        mae += abs(den_pred.sum()-den_gt.sum())
        mse += (den_pred.sum()-den_gt.sum())**2

    mae = mae/len(val_loader)
    mse = np.sqrt(mse/len(val_loader))
    logging.info(' * MAE {:.3f}'.format(mae))
    logging.info(' * MSE {:.3f}'.format(mse))

    return mae, mse


def validate_seg(opt, val_loader, net):
    print ('begin test seg')
    net.eval()
    
    pa_1, pa_2 = 0, 0
    cpa_1, cpa_2 = 0, 0
    mpa_1, mpa_2 = 0, 0
    IoU_1, IoU_2 = 0, 0
    mIoU_1, mIoU_2 = 0, 0

    for i,(img, den_target, _, _, _) in enumerate(val_loader):
        img = img.cuda()

        den_output, seg_output_1, seg_output_2,_, _= net(img)
        seg_target_1 = den_to_seg(den_target, threshold=0.0)
        _, max_idx_1 = torch.max(seg_output_1, dim=1)
        max_idx_1 = max_idx_1.squeeze(0)
        seg_target_1 = seg_target_1.data.cpu().numpy()
        max_idx_1 = max_idx_1.data.cpu().numpy()
        metric_1 = SegmentationMetric(2)
        hist = metric_1.addBatch(max_idx_1, seg_target_1)
        pa_1 += metric_1.pixelAccuracy()
        cpa_1 += metric_1.classPixelAccuracy()
        mpa_1 += metric_1.meanPixelAccuracy()
        IoU_1 += metric_1.IntersectionOverUnion()
        mIoU_1 += metric_1.meanIntersectionOverUnion()


        seg_target_2 = den_to_seg(den_target, threshold=0.0)
        seg_target_2 = seg_target_2.data.cpu().numpy()
        _, max_idx_2 = torch.max(seg_output_2, dim=1)
        max_idx_2 = max_idx_2.squeeze(0)
        max_idx_2 = max_idx_2.data.cpu().numpy()
        metric_2 = SegmentationMetric(2)
        hist = metric_2.addBatch(max_idx_2, seg_target_2)
        pa_2 += metric_2.pixelAccuracy()
        cpa_2 += metric_2.classPixelAccuracy()
        mpa_2 += metric_2.meanPixelAccuracy()
        IoU_2 += metric_2.IntersectionOverUnion()
        mIoU_2 += metric_2.meanIntersectionOverUnion()

    pa_1, pa_2 = pa_1/len(val_loader), pa_2/len(val_loader)
    cpa_1, cpa_2 = cpa_1/len(val_loader), cpa_2/len(val_loader)
    mpa_1, mpa_2 = mpa_1/len(val_loader), mpa_2/len(val_loader)
    IoU_1, IoU_2 = IoU_1/len(val_loader), IoU_2/len(val_loader)
    mIoU_1, mIoU_2 = mIoU_1/len(val_loader), mIoU_2/len(val_loader)

    print('PA_1 is : %f' % pa_1)
    print('cPA_1 is :', cpa_1)
    print('mPA_1 is : %f' % mpa_1)
    print('IoU_1 is : ', IoU_1)
    print('mIoU_1 is : ', mIoU_1)
    print('\n')
    print('PA_2 is : %f' % pa_2)
    print('cPA_2 is :', cpa_2)
    print('mPA_2 is : %f' % mpa_2)
    print('IoU_2 is : ', IoU_2)
    print('mIoU_2 is : ', mIoU_2)

    return mIoU_1, mIoU_2
