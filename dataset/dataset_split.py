import os
import h5py
import numpy as np
import glob
from PIL import Image

if __name__ == "__main__":
    data_name = 'SHB'
    label_rate = 0

    if data_name == 'SHA':
        train_path = '/root/CUPCC/Dataset/SHA/Train/images'
        test_path = '/root/CUPCC/Dataset/SHA/Test/images'
        label_rate = 0.2
        # label_rate = 0.5
    elif data_name == 'SHB':
        train_path = '/root/CUPCC/Dataset/SHB/Train/images'
        test_path = '/root/CUPCC/Dataset/SHB/Test/images'
        label_rate = 0.05
    elif data_name == 'UCF-QNRF_ECCV18':
        train_path = '/home/linkdata/data/lijingqing/CSRNet/dataset/UCF-QNRF_ECCV18/Train/images'
        test_path = '/home/linkdata/data/lijingqing/CSRNet/dataset/UCF-QNRF_ECCV18/Test/images'
        label_rate = 0.2
    elif data_name == 'NWPU':
        train_path = '/home/linkdata/data/lijingqing/CSRNet/dataset/NWPU/Train/images'
        test_path = '/home/linkdata/data/lijingqing/CSRNet/dataset/NWPU/Test/images'
        label_rate = 0.1
    elif data_name == 'JHU':
        train_path = '/home/linkdata/data/lijingqing/CUPCC/Dataset/JHU/train/images'
        val_path = '/home/linkdata/data/lijingqing/CUPCC/Dataset/JHU/val/images'
        test_path = '/home/linkdata/data/lijingqing/CUPCC/Dataset/JHU/test/images'
        label_rate = 0.5
        name_val = 'val'


    name_train_all = 'train_all'
    name_train_label = 'train_label_5%'
    name_train_unlabl = 'train_unlabel_5%'
    name_test = 'test'

    if data_name == 'UCF-QNRF_ECCV18' or data_name == 'NWPU':
        train_files = glob.glob(train_path + '/*.jpg')
        test_files = glob.glob(test_path + '/*.jpg')
    elif data_name == 'SHA' or data_name == 'SHB':
        train_files = os.listdir(train_path)
        test_files = os.listdir(test_path)

    train_num = len(train_files)
    test_num = len(test_files)

    m = open(os.path.join(os.path.abspath('.'), data_name, data_name + '_' + name_train_all + '.json'), 'w', encoding='utf-8')
    f = open(os.path.join(os.path.abspath('.'), data_name, data_name + '_' + name_train_label + '.json'), 'w', encoding='utf-8')
    h = open(os.path.join(os.path.abspath('.'), data_name, data_name + '_' + name_train_unlabl + '.json'), 'w', encoding='utf-8')
    f.write('[')
    h.write('[')
    m.write('[')

    for i, file in enumerate(train_files):

        img = Image.open(os.path.join(train_path, file))
        if i<round(train_num*label_rate)-1:
            m.write('"' + os.path.join(train_path, file) + '",\n')
            f.write('"' + os.path.join(train_path, file) + '",\n')
        elif i==round(train_num*label_rate)-1:
            m.write('"' + os.path.join(train_path, file) + '",\n')
            f.write('"' + os.path.join(train_path, file) + '"]')
        elif i>round(train_num*label_rate)-1 and i<round(train_num-1):
            m.write('"' + os.path.join(train_path, file) + '",\n')
            h.write('"' + os.path.join(train_path, file) + '",\n')
        elif i==round(train_num-1):
            m.write('"' + os.path.join(train_path, file) + '"]')
            h.write('"' + os.path.join(train_path, file) + '"]')

    m.close()
    h.close()
    f.close()


    f = open(os.path.join(os.path.abspath('.'), data_name, data_name + '_' + name_test + '.json'), 'w', encoding='utf-8')
    f.write('[')
    for i, file in enumerate(test_files):
        img = Image.open(os.path.join(train_path, file))
        if i<round(test_num-1):
            f.write('"' + os.path.join(test_path, file) + '",\n')
        elif i==round(test_num-1):
            f.write('"' + os.path.join(test_path, file) + '"]')
    f.close()