import os
import json
import cv2
import numpy as np
import pickle
import pydicom
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from .utils_3d import *


class Pan_dataset_3d(Dataset):
    def __init__(self, args, mode='train') -> None:
        super(Pan_dataset_3d, self).__init__()
        self.args = args
        self.split_info = json.load(open(args.split_info,'r'))
        self.mode = mode
        if mode in ['train']:
            self.item_list = self.split_info['train']
        elif mode in ['val']:
            self.item_list = self.split_info['val']
        elif mode in ['all']:
            self.item_list = self.split_info['train'] + self.split_info['val']
        elif mode in ['test']:
            self.item_list = self.split_info['test']
        elif mode in ['test_train']:
            self.item_list = self.split_info['train'] + self.split_info['test']
        elif mode in ['test_all']:
            self.item_list = self.split_info['train'] + self.split_info['val'] + self.split_info['test']
        self.anno_dict = json.load(open(args.cls_anno_path,'r', encoding='utf-8'))
        
    def __getitem__(self, index):
        item_dict = self.item_list[index]
        anno = self.anno_dict[item_dict['anno_item']]
        if self.args.data_mode.lower() in ['img', 'image']:
            img, label = read_img_n_label_files(self.args, item_dict, self.mode)    # CxDxHxW
        elif self.args.data_mode.lower() in ['dcm', 'dicom']:
            if self.args.selected_window:
                img, label = read_dcm_n_label_files_win(self.args, item_dict, self.mode, anno)
            elif self.args.D_soft_sample:
                img, label, info_dict = read_dcm_n_label_files_soft(self.args, item_dict, self.mode, anno)
            else:
                img, label = read_dcm_n_label_files(self.args, item_dict, self.mode)
        elif self.args.data_mode.lower() in ['img_dat', 'image_dat']:
            img, label = read_img_n_label_dat_files(self.args, item_dict, self.mode, anno)
        elif self.args.data_mode.lower() in ['dcm_dat', 'dicom_dat']:
            img, label = read_dcm_n_label_dat_files(self.args, item_dict, self.mode)
        else:
            raise NotImplementedError("Data mode: '{}' is not implemented!")
        surgery_label = anno['surgery_label']
        invade_label = anno['invade_label']
        surgery_gt = anno['surgery_label']
        invade_gt = anno['invade_label']
        if self.args.net_invade_classes == 2:
            invade_label = 1 if invade_label>=1 else 0
        
        if self.args.selected_window:
            if self.mode in ['train', 'all']:
                img, label = random_crop(self.args, img, label, anno)
            elif self.mode in ['val', 'test']:
                img, label = center_crop(self.args, img, label)
            label[label==128] = 1
            label[label==255] = 2
            surgery_label = torch.tensor(surgery_label)
            invade_label = torch.tensor(invade_label)
        elif self.args.D_soft_sample:
            if self.mode in ['train', 'all']:
                img, label, info_dict = random_crop_soft(self.args, img, label, info_dict)
            elif self.mode in ['val', 'test']:
                if self.args.slide_window:
                    img, label, info_dict = center_crop_soft(self.args, img, label, info_dict)
                else:
                    img, label = center_crop(self.args, img, label)
            label[label==128] = 1
            label[label==255] = 2
            surgery_label = torch.tensor([1-surgery_label*info_dict['iou'], surgery_label*info_dict['iou']])
            invade_label = torch.tensor([1-invade_label*info_dict['iou'], invade_label*info_dict['iou']])
        else:
            if self.mode in ['train', 'all']:
                if self.args.rand_rotate:
                    img, label = random_rotate(self.args, img, label)
                if self.args.rand_scale:
                    img, label = random_scale(self.args, img, label)
                # Random Crop
                if self.args.window_crop:
                    img, label = random_crop_with_window(self.args, img, label)
                else:
                    img, label = random_crop(self.args, img, label, anno)
            elif self.mode in ['val', 'test']:
                img, label = center_crop(self.args, img, label, anno)
            label[label==128] = 1
            label[label==255] = 2
            surgery_label = torch.tensor(surgery_label)
            invade_label = torch.tensor(invade_label)
        
        info_meta = read_img_meta(anno)
        if self.args.rand_blood_text:
            info_meta['blood_des'] = random_blood_text(info_meta['blood_des'])
        if self.args.rand_others_text:
            info_meta['others_des'] = random_others_text(info_meta['others_des'])
            
        img_meta = {
            "patient_id": info_meta['patient_id'],
            'gt_surgery': surgery_gt,
            'gt_invade': invade_gt,
            'label_surgery': surgery_label,
            'label_invade': invade_label,
            'anno_item': item_dict['anno_item'],
            "blood": info_meta['blood'],
            "others": info_meta['others'],
            "blood_des": info_meta['blood_des'],
            "others_des": info_meta['others_des'],
        }
        return img, label, img_meta
    
    def __len__(self):
        return len(self.item_list)
    
