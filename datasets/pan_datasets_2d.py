import os
import json
import cv2
import random
import numpy as np
import pickle
import pydicom
import torch
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from .utils_2d import *

class Pan_dataset_2d(Dataset):
    def __init__(self, args, mode='train') -> None:
        super(Pan_dataset_2d, self).__init__()
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
        elif mode in ['val_test']:
            self.item_list = self.split_info['val'] + self.split_info['test']
        self.anno_dict = json.load(open(args.cls_anno_path,'r'))
        if args.use_detection_slice:
            assert args.net_essential_object.lower() in ['tumor']
            self.item_list = recollect_item(self.mode, self.item_list, self.anno_dict)
        
        self.esse_list = get_esse_list(args, self.item_list, self.anno_dict)
        
    def __getitem__(self, index):
        item_dict = self.item_list[index]
        if self.args.data_mode.lower() in ['img', 'image', 'png']:
            img, label = read_img_n_label_files(self.args, item_dict, self.mode)
        elif self.args.data_mode.lower() in ['dcm', 'dicom']:
            img, label = read_dcm_n_label_files(self.args, item_dict, self.mode)
        else:
            raise NotImplementedError("Data mode: '{}' is not implemented!")
        
        img_name = os.path.basename(item_dict['image_path']).split('.')[0]
        anno = self.anno_dict[item_dict['anno_item']]
        surgery_label = anno['surgery_label']
        invade_label = anno['invade_label']
        essential_label = get_essential_label(self.args, item_dict, anno)
        
        if self.args.use_detection:
            img, label = detection_crop(self.args, img, label, anno)
        
        if self.mode in ['train', 'all']:
            # Resize
            img, label = resize(self.args, img, label)
            # Random Crop
            if self.args.window_crop:
                img, label = random_crop_with_window(self.args, img, label)
            else:
                img, label = random_crop(self.args, img, label)
        elif self.mode in ['val', 'test']:
            # Resize
            img, label = resize(self.args, img, label)
            # Center Crop
            img, label = center_crop(self.args, img, label)
        label[label==128] = 1
        label[label==255] = 2
        
        img = ToTensor()(img)
        label = torch.tensor(label)
        surgery_label = torch.tensor(surgery_label)
        invade_label = torch.tensor(invade_label)
        essential_label = torch.tensor(essential_label)
        
        info_meta = read_img_meta(self.args, anno, self.mode)
            
        img_meta = {
            "patient_id": info_meta['patient_id'],
            "img_name": img_name,
            'label_surgery': surgery_label,
            'label_invade': invade_label,
            'label_essential': essential_label,
            'anno_item': item_dict['anno_item'],
            "blood": info_meta['blood'],
            "others": info_meta['others'],
            "blood_des": info_meta['blood_des'],
            "others_des": info_meta['others_des'],
        }
        return img, label, img_meta
    
    def __len__(self):
        return len(self.item_list)


class EqualSampler(Sampler):
    def __init__(self, data_source:Pan_dataset_2d, batch_size) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.esse_list = data_source.esse_list
        self.final_idx = self._get_final_idx()
    
    def _get_final_idx(self):
        assert self.batch_size%2 == 0
        idx_esse = np.where(np.array(self.esse_list)==1)[0].tolist()
        idx_non_esse = np.where(np.array(self.esse_list)==0)[0].tolist()
        sample_num = min(len(idx_esse), len(idx_non_esse))
        idx_esse = idx_esse[:sample_num]
        idx_non_esse = idx_non_esse[:sample_num]
        dropped = len(idx_esse)%int(self.batch_size/2)
        idx_esse = idx_esse[:-dropped]
        idx_non_esse = idx_non_esse[:-dropped]
        final_idx = []
        for i in range(int(len(idx_esse)/int(self.batch_size/2))):
            sampled_esse = random.sample(idx_esse, int(self.batch_size/2))
            sampled_non_esse = random.sample(idx_non_esse, int(self.batch_size/2))
            for sample_idx in sampled_esse:
                idx_esse.remove(sample_idx)
            for sample_idx in sampled_non_esse:
                idx_non_esse.remove(sample_idx)
                
            sampled = sampled_esse+sampled_non_esse
            random.shuffle(sampled)
            final_idx.extend(sampled)
        return final_idx

    def __iter__(self):
        return iter(self._get_final_idx())

    def __len__(self):
        return len(self.final_idx)

def find_indices(lst, value):
    indices = []
    for i in range(len(lst)):
        if lst[i] == value:
            indices.append(i)
    return indices

def get_class_weighted_indice(args, dataset: Pan_dataset_2d, weights):
    weights = [w/sum(weights) for w in weights]
    item_list = dataset.item_list
    anno_dict = dataset.anno_dict
    cls_list = [get_essential_label(args, item_dict, anno_dict[item_dict['anno_item']]) for item_dict in item_list]
    esse_indice = find_indices(cls_list,1)
    nonesse_indice = find_indices(cls_list,0)
    nonesse_num = round((len(esse_indice)/weights[1])*weights[0])
    select_nonesse_indice = random.sample(nonesse_indice, nonesse_num)
    indice_list = sorted(esse_indice + select_nonesse_indice)
    
    return indice_list
