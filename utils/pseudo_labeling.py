import torch
import torch.nn.functional as F
from utils.utils import *
from tqdm import tqdm
import time
from torch.autograd import Variable
from torchvision import datasets, transforms


def pseudo_labeling(data_loader, net):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    f_pass = 5
    pseudo_den_list = []
    gt_den_list = []
    uncertainty_den_list = []
    weight_uncertainty_den_list = []
    pseudo_seg_list = []
    uncertainty_seg_list = []
    pseudo_seg_1_list = [] 
    uncertainty_seg_1_list = []
    pseudo_seg_2_list = [] 
    uncertainty_seg_2_list = []
    pseudo_seg_3_list = [] 
    uncertainty_seg_3_list = []

    gt_seg_list = []


    pseudo_img_path_list = []

    data_loader = tqdm(data_loader)

    with torch.no_grad():
        for batch_idx, (img, den_target, _, img_path, flag) in enumerate(data_loader):
            data_time.update(time.time() - end)

            img_path = img_path[0]

            img = img.cuda()
            img = Variable(img)
            seg_target = den_to_seg(den_target, 0)

            out_den_list = []
            out_seg_list_1 = []
            out_seg_list_2 = []
            out_seg_list_3 = []
 
            for _ in range(f_pass):

                img = img + torch.clamp(torch.randn_like(img) * 0.1, -0.2, 0.2)
                # den_output,  seg_output_1, seg_output_2, seg_output_3 = net(img)
                den_output,  seg_output_1, seg_output_2 = net(img)
                out_den_list.append(den_output.squeeze(0).squeeze(0))
                out_seg_list_1.append(F.softmax(seg_output_1, dim=1))
                out_seg_list_2.append(F.softmax(seg_output_2, dim=1))
                # out_seg_list_3.append(F.softmax(seg_output_3, dim=1))

            out_den_list = torch.stack(out_den_list)
            out_seg_list_1 = torch.stack(out_seg_list_1)
            out_seg_list_2 = torch.stack(out_seg_list_2)
            # out_seg_list_3 = torch.stack(out_seg_list_3)


            out_den_sum = torch.sum(out_den_list, dim=(1,2))
            out_den_std = torch.std(out_den_list, dim=0)
            out_den_mean = torch.mean(out_den_list, dim=0)
            out_den_sum_mean = torch.sum(out_den_mean, dim=(0,1))
            max_den_std = out_den_std

            uncertainty_den = max_den_std.cpu().numpy()
            out_den_mean = out_den_mean.data.cpu().numpy().tolist()

            uncertainty_den_max = np.max(uncertainty_den)
            uncertainty_den_min = np.min(uncertainty_den)
            weight_uncertainty_den = 1-(uncertainty_den -uncertainty_den_min)/(uncertainty_den_max-uncertainty_den_min) 

            

            out_seg_mean_1 = torch.mean(out_seg_list_1, dim=0)
            out_seg_mean_2 = torch.mean(out_seg_list_2, dim=0) 
            # out_seg_mean_3 = torch.mean(out_seg_list_3, dim=0)

            max_value_1, max_idx_1 = torch.max(out_seg_mean_1, dim=1)
            max_value_2, max_idx_2 = torch.max(out_seg_mean_2, dim=1)
            # max_value_3, max_idx_3 = torch.max(out_seg_mean_3, dim=1)

            out_seg_std_1 = torch.std(out_seg_list_1, dim=0)
            out_seg_std_2 = torch.std(out_seg_list_2, dim=0)
            # out_seg_std_3 = torch.std(out_seg_list_3, dim=0)

            max_seg_std_1 = out_seg_std_1[:,0]  # [:,0]==[:,1]
            max_seg_std_2 = out_seg_std_2[:,0]
            # max_seg_std_3 = out_seg_std_3[:,0]

            
            # max_seg_std_sort = max_seg_std_1.reshape(-1).sort(descending=True)[0]
            # seg_uncer_th1 = max_seg_std_sort[0]
            # seg_uncer_th2 = max_seg_std_sort[int(0.5*len(max_seg_std_sort))]
            # seg_uncer_th3 = max_seg_std_sort[int(0.7*len(max_seg_std_sort))]

            uncertainty_seg_1 = max_seg_std_1.cpu().numpy().squeeze(0)
            uncertainty_seg_2 = max_seg_std_2.cpu().numpy().squeeze(0)
            # uncertainty_seg_3 = max_seg_std_3.cpu().numpy().squeeze(0)

            # pseudo_seg_fore_idx = torch.where(max_idx_1==1)
            # max_seg_std_fore_1 = max_seg_std_1[pseudo_seg_fore_idx]
            # max_seg_std_fore_1_sort = max_seg_std_fore_1.reshape(-1).sort()[0]
            # max_seg_std_fore_2 = max_seg_std_2[pseudo_seg_fore_idx]
            # max_seg_std_fore_2_sort = max_seg_std_fore_2.reshape(-1).sort()[0]
            # max_seg_std_fore_3 = max_seg_std_3[pseudo_seg_fore_idx]
            # max_seg_std_fore_3_sort = max_seg_std_fore_3.reshape(-1).sort()[0]
            # seg_fore_uncer_th1 = max_seg_std_fore_1_sort[int(1*len(max_seg_std_fore_1_sort))-1] if len(max_seg_std_fore_1_sort)>0 else 0
            # seg_fore_uncer_th2 = max_seg_std_fore_2_sort[int(1*len(max_seg_std_fore_2_sort))-1] if len(max_seg_std_fore_2_sort)>0 else 0
            # seg_fore_uncer_th3 = max_seg_std_fore_3_sort[int(0.3*len(max_seg_std_fore_3_sort))] if len(max_seg_std_fore_3_sort)>0 else 0
            # seg_fore_selected_idx_1 = (max_value_1>=0.49) * (max_idx_1==1) * (max_seg_std_1.squeeze(1)<=seg_fore_uncer_th1) # 置信度分数高且不确定性度低, 认为是正确标签
            # seg_fore_selected_idx_2 = (max_value_2>=0.49) * (max_idx_1==1) * (max_seg_std_2.squeeze(1)<=seg_fore_uncer_th2)
            # seg_fore_selected_idx_3 = (max_value_3>=0.49) * (max_idx_1==1) * (max_seg_std_3.squeeze(1)<=seg_fore_uncer_th3)

            # pseudo_seg_back_idx = torch.where(max_idx_1==1)
            # max_seg_std_back_1 = max_seg_std_1[pseudo_seg_back_idx]
            # max_seg_std_back_1_sort = max_seg_std_back_1.reshape(-1).sort()[0]
            # max_seg_std_back_2 = max_seg_std_2[pseudo_seg_back_idx]
            # max_seg_std_back_2_sort = max_seg_std_back_2.reshape(-1).sort()[0]
            # max_seg_std_back_3 = max_seg_std_3[pseudo_seg_back_idx]
            # max_seg_std_back_3_sort = max_seg_std_back_3.reshape(-1).sort()[0]
            # seg_back_uncer_th1 = max_seg_std_back_1_sort[int(1*len(max_seg_std_back_1_sort))-1] if len(max_seg_std_back_1_sort)>0 else 0
            # seg_back_uncer_th2 = max_seg_std_back_2_sort[int(1*len(max_seg_std_back_2_sort))-1] if len(max_seg_std_back_2_sort)>0 else 0
            # seg_back_uncer_th3 = max_seg_std_back_3_sort[int(0.3*len(max_seg_std_back_3_sort))] if len(max_seg_std_back_3_sort)>0 else 0
            # seg_back_selected_idx_1 = (max_value_1>=0.49) * (max_idx_1==0) * (max_seg_std_1.squeeze(1)<=seg_back_uncer_th1) # 置信度分数高且不确定性度低, 认为是正确标签
            # seg_back_selected_idx_2 = (max_value_2>=0.49) * (max_idx_1==0) * (max_seg_std_2.squeeze(1)<=seg_back_uncer_th2)
            # seg_back_selected_idx_3 = (max_value_3>=0.49) * (max_idx_1==0) * (max_seg_std_3.squeeze(1)<=seg_back_uncer_th3) 

            # pseudo_seg_1 = (torch.ones_like(seg_fore_selected_idx_1) * 10).type(torch.LongTensor)
            # pseudo_seg_1.index_put_([seg_fore_selected_idx_1], torch.tensor(1.).type(torch.LongTensor))
            # pseudo_seg_1.index_put_([seg_back_selected_idx_1], torch.tensor(0.).type(torch.LongTensor))

            # pseudo_seg_2 = (torch.ones_like(seg_fore_selected_idx_2) * 10).type(torch.LongTensor)
            # pseudo_seg_2.index_put_([seg_fore_selected_idx_2], torch.tensor(1.).type(torch.LongTensor))
            # pseudo_seg_2.index_put_([seg_back_selected_idx_2], torch.tensor(0.).type(torch.LongTensor)) 

            # pseudo_seg_3 = (torch.ones_like(seg_fore_selected_idx_3) * 10).type(torch.LongTensor)
            # pseudo_seg_3.index_put_([seg_fore_selected_idx_3], torch.tensor(1.).type(torch.LongTensor))
            # pseudo_seg_3.index_put_([seg_back_selected_idx_3], torch.tensor(0.).type(torch.LongTensor))

            # pseudo_seg_1 = pseudo_seg_1.squeeze(0).cpu().numpy().tolist()
            # pseudo_seg_2 = pseudo_seg_2.squeeze(0).cpu().numpy().tolist()
            # pseudo_seg_3 = pseudo_seg_3.squeeze(0).cpu().numpy().tolist()

            seg_selected_idx_1 = (max_value_1>=0.49) * (max_seg_std_1.squeeze(1)<=1000) # 置信度分数高且不确定性度低, 认为是正确标签
            seg_selected_idx_2 = (max_value_2>=0.49) * (max_seg_std_2.squeeze(1)<=1000)
            # seg_selected_idx_3 = (max_value_3>=0.49) * (max_seg_std_3.squeeze(1)<=1000) 

            pseudo_seg_1_1 = torch.where(seg_selected_idx_1, 0, 10).squeeze(0).cpu().numpy() # 10 代表不可信
            pseudo_seg_1_0 = torch.where(max_idx_1==0, 0, 1).squeeze(0).cpu().numpy()
            pseudo_seg_1 = (pseudo_seg_1_1 + pseudo_seg_1_0).clip(0,10).tolist()
            
            pseudo_seg_2_1 = torch.where(seg_selected_idx_2, 0, 10).squeeze(0).cpu().numpy() # 10 代表不可信
            pseudo_seg_2_0 = torch.where(max_idx_2==0, 0, 1).squeeze(0).cpu().numpy()
            pseudo_seg_2 = (pseudo_seg_2_1 + pseudo_seg_2_0).clip(0,10).tolist()

            # pseudo_seg_3_1 = torch.where(seg_selected_idx_3, 0, 10).squeeze(0).cpu().numpy() # 10 代表不可信
            # pseudo_seg_3_0 = torch.where(max_idx_3==0, 0, 1).squeeze(0).cpu().numpy()
            # pseudo_seg_3 = (pseudo_seg_3_1 + pseudo_seg_3_0).clip(0,10).tolist()
    
 
            pseudo_img_path = str(img_path)
            den_target = den_target.squeeze(0).data.cpu().numpy().tolist()
            gt_seg = seg_target.cpu().numpy().tolist()

            data_loader.set_description("Pseudo-Labeling Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s".
                                        format(batch=batch_idx + 1,
                                                iter=len(data_loader),
                                                data=data_time.avg,
                                                bt=batch_time.avg))



            pseudo_img_path_list.append(pseudo_img_path)

            pseudo_den_list.append(out_den_mean)
            gt_den_list.append(den_target)
            uncertainty_den_list.append(uncertainty_den)
            weight_uncertainty_den_list.append(weight_uncertainty_den)

            gt_seg_list.append(gt_seg)
            pseudo_seg_1_list.append(pseudo_seg_1)
            pseudo_seg_2_list.append(pseudo_seg_2)
            # pseudo_seg_3_list.append(pseudo_seg_3)
         
            uncertainty_seg_1_list.append(uncertainty_seg_1)
            uncertainty_seg_2_list.append(uncertainty_seg_2)
            # uncertainty_seg_3_list.append(uncertainty_seg_3)

            pseudo_seg_tmp = []
            pseudo_seg_tmp.append(pseudo_seg_1)
            pseudo_seg_tmp.append(pseudo_seg_2)
            # pseudo_seg_tmp.append(pseudo_seg_3)
            pseudo_seg_list.append(pseudo_seg_tmp)

            uncertainty_seg_tmp = []
            uncertainty_seg_tmp.append(uncertainty_seg_1)
            uncertainty_seg_tmp.append(uncertainty_seg_2)
            # uncertainty_seg_tmp.append(uncertainty_seg_3)
            uncertainty_seg_list.append(uncertainty_seg_tmp)


        data_loader.close()


        pseudo_img_path_list = np.array(pseudo_img_path_list)
        pseudo_den_list = np.array(pseudo_den_list)
        gt_den_list = np.array(gt_den_list)
        uncertainty_den_list = np.array(uncertainty_den_list)
        weight_uncertainty_den_list = np.array(weight_uncertainty_den_list)

        gt_seg_list = np.array(gt_seg_list)
        pseudo_seg_1_list = np.array(pseudo_seg_1_list)  # N,H,W
        pseudo_seg_2_list = np.array(pseudo_seg_2_list)
        # pseudo_seg_3_list = np.array(pseudo_seg_3_list) 
        uncertainty_seg_1_list = np.array(uncertainty_seg_1_list)
        uncertainty_seg_2_list = np.array(uncertainty_seg_2_list)
        uncertainty_seg_3_list = np.array(uncertainty_seg_3_list)

        pseudo_label_dict = {'pseudo_img_path': pseudo_img_path_list,
                'pseudo_den_list': pseudo_den_list,
                'gt_den_list': gt_den_list,
                'uncertainty_den_list': uncertainty_den_list,
                'weight_uncertainty_den_list': weight_uncertainty_den_list,
                'gt_seg_list': gt_seg_list,
                'pseudo_seg_list': pseudo_seg_list,
                'uncertainty_seg_list': uncertainty_seg_list}

        # acc_seg(gt_seg_list, pseudo_seg_1_list)
        acc_seg_back(gt_seg_list, pseudo_seg_1_list)
        acc_seg_back(gt_seg_list, pseudo_seg_2_list)
        # acc_seg(gt_seg_list, pseudo_seg_3_list)

        return pseudo_label_dict


def acc_seg(gt_seg_list, pseudo_seg_list):

    pseudo_seg = []
    gt_seg = []
    
    for idx in range(len(pseudo_seg_list)):
        pseudo_seg_arr = np.array(pseudo_seg_list[idx]).reshape(-1)
        pseudo_seg.extend(pseudo_seg_arr)
        gt_seg_arr = np.array(gt_seg_list[idx]).reshape(-1)
        gt_seg.extend(gt_seg_arr)


    pseudo_seg = np.array(pseudo_seg)
    gt_seg = np.array(gt_seg)
    

    pseudo_seg_nums_0 = ((gt_seg==0)*1).sum()
    pseudo_seg_select_0 = ((pseudo_seg==0)*1).sum()
    pseudo_seg_correct_0 = (((pseudo_seg==0)*(gt_seg==0))*1).sum()
    pseudo_seg_select_per_0 = (pseudo_seg_select_0/pseudo_seg_nums_0)*100
    pseudo_seg_correct_acc_0 = (pseudo_seg_correct_0/pseudo_seg_select_0)*100
    print(f'pseudo-seg background select nums: {pseudo_seg_select_0}, correct nums: {pseudo_seg_correct_0}, gt nums: {pseudo_seg_nums_0}')
    print(f'pseudo-seg background select per: {pseudo_seg_select_per_0}, correct per: {pseudo_seg_correct_acc_0}')

    pseudo_seg_nums_1 = ((gt_seg==1)*1).sum()
    pseudo_seg_select_1 = ((pseudo_seg==1)*1).sum()
    pseudo_seg_correct_1 = (((pseudo_seg==1)*(gt_seg==1))*1).sum()
    pseudo_seg_select_per_1 = (pseudo_seg_select_1/pseudo_seg_nums_1)*100
    pseudo_seg_correct_acc_1 = (pseudo_seg_correct_1/pseudo_seg_select_1)*100
    print(f'pseudo-seg foreground select nums: {pseudo_seg_select_1}, correct nums: {pseudo_seg_correct_1}, gt nums: {pseudo_seg_nums_1}')
    print(f'pseudo-seg foreground select per: {pseudo_seg_select_per_1}, correct per: {pseudo_seg_correct_acc_1}')

    pseudo_seg_select = (pseudo_seg != 10)*1
    pseudo_seg_correct = (pseudo_seg == gt_seg)*1
    pseudo_seg_select_per = (sum(pseudo_seg_select)/len(pseudo_seg.reshape(-1)))*100
    pseudo_seg_correct_acc = (sum(pseudo_seg_correct)/sum(pseudo_seg_select))*100
    print(f'pseudo-seg select per: {pseudo_seg_select_per}, correct per: {pseudo_seg_correct_acc} \n')
    print('-----------------------------------------------------------------------------')


def acc_seg_back(gt_seg_list, pseudo_seg_list):
    pseudo_seg = []
    gt_seg = []
    
    for idx in range(len(pseudo_seg_list)):
        pseudo_seg_arr = np.array(pseudo_seg_list[idx]).reshape(-1)
        pseudo_seg.extend(pseudo_seg_arr)
        gt_seg_arr = np.array(gt_seg_list[idx]).reshape(-1)
        gt_seg.extend(gt_seg_arr)


    pseudo_seg = np.array(pseudo_seg)
    gt_seg = np.array(gt_seg)
    
    pseudo_seg_nums_0 = ((gt_seg==0)*1).sum()
    pseudo_seg_select_0 = ((pseudo_seg==1)*1).sum()
    pseudo_seg_correct_0 = (((pseudo_seg==1)*(gt_seg==0))*1).sum()
    pseudo_seg_select_per_0 = (pseudo_seg_select_0/pseudo_seg_nums_0)*100
    pseudo_seg_correct_acc_0 = (pseudo_seg_correct_0/pseudo_seg_select_0)*100
    print(f'pseudo-seg background select nums: {pseudo_seg_select_0}, correct nums: {pseudo_seg_correct_0}, gt nums: {pseudo_seg_nums_0}')
    print(f'pseudo-seg background select per: {pseudo_seg_select_per_0}, correct per: {pseudo_seg_correct_acc_0}')

    pseudo_seg_nums_1 = ((gt_seg==1)*1).sum()
    pseudo_seg_select_1 = ((pseudo_seg==0)*1).sum()
    pseudo_seg_correct_1 = (((pseudo_seg==0)*(gt_seg==1))*1).sum()
    pseudo_seg_select_per_1 = (pseudo_seg_select_1/pseudo_seg_nums_1)*100
    pseudo_seg_correct_acc_1 = (pseudo_seg_correct_1/pseudo_seg_select_1)*100
    print(f'pseudo-seg foreground select nums: {pseudo_seg_select_1}, correct nums: {pseudo_seg_correct_1}, gt nums: {pseudo_seg_nums_1}')
    print(f'pseudo-seg foreground select per: {pseudo_seg_select_per_1}, correct per: {pseudo_seg_correct_acc_1}')

    pseudo_seg_select = (pseudo_seg != 10)*1
    pseudo_seg_correct = pseudo_seg_correct_0 +pseudo_seg_correct_1
    pseudo_seg_select_per = (sum(pseudo_seg_select)/len(pseudo_seg.reshape(-1)))*100
    pseudo_seg_correct_acc = (pseudo_seg_correct/sum(pseudo_seg_select))*100
    print(f'pseudo-seg select per: {pseudo_seg_select_per}, correct per: {pseudo_seg_correct_acc} \n')
    print('-----------------------------------------------------------------------------')


    



     