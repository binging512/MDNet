import os
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torchvision.transforms import ColorJitter
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from datasets.pan_datasets_2d import Pan_dataset_2d, get_class_weighted_indice

from models import Custom_Model_2D
from models.img_encoder_2d.UNet2d import UNet
import models.text_encoder.clip as clip
from models.loss import *

from utils.eval_2d import vote_w_esse, save_results
from utils.cal_metrics import cal_class_metrics

from IPython.core.debugger import set_trace

def test_2d(args, logger):
    test_dataset = Pan_dataset_2d(args, mode = 'test')
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
    model = Custom_Model_2D(args).cuda()
    
    model.load_state_dict(torch.load(os.path.join(args.workspace, args.test_checkpoint)))
    model.eval()
    
    logger.info("============== Testing ==============")
    pred_invade_dict, pred_surgery_dict, pred_essential_dict = {}, {}, {}
    score_invade_dict, score_surgery_dict, score_essential_dict = {}, {}, {}
    label_invade_dict, label_surgery_dict, label_essential_dict = {}, {}, {}
    feat_dict = {}
    
    for ii, item in enumerate(dataloader):
        if ii%100 == 0:
            logger.info("Validating the {}/{} images...".format(ii,len(dataloader)))
        img, label, img_meta = item
        
        img = img.cuda()
        anno_items = img_meta['anno_item']
        img_names = img_meta['img_name']
        
        input_batch = {
            'img': img,
            "blood": img_meta['blood'].cuda(),
            'others': img_meta['others'].cuda(),
            "blood_des": clip.tokenize(img_meta['blood_des'], context_length=256).cuda(),
            "others_des": clip.tokenize(img_meta['others_des']).cuda(),
        }
        
        with torch.no_grad():
            output = model(input_batch)
            preds_invade = output['preds_invade']
            preds_surgery = output['preds_surgery']
            preds_essential = output['preds_essential']
            preds_seg = output['pred_seg']
            feat = output['feat'].detach().cpu().numpy()
            # preds_invade, preds_surgery, preds_essential, preds_seg = model(img)
                
        # Fusion
        if args.test_fusion =='mean':
            pred_invade = torch.mean(torch.stack(preds_invade, dim=0), dim=0)
            pred_surgery = torch.mean(torch.stack(preds_surgery, dim=0), dim=0)
            pred_essential = torch.mean(torch.stack(preds_essential, dim=0), dim=0)
        if args.test_fusion == 'max':
            pred_invade,_ = torch.max(torch.stack(preds_invade, dim=0), dim=0)
            pred_surgery,_ = torch.max(torch.stack(preds_surgery, dim=0), dim=0)
            pred_essential,_ = torch.max(torch.stack(preds_essential, dim=0), dim=0)
            
        pred_invade = torch.softmax(pred_invade.detach().cpu(), dim=1)
        pred_surgery = torch.softmax(pred_surgery.detach().cpu(), dim=1)
        pred_essential = torch.softmax(pred_essential.detach().cpu(), dim=1)

        pred_invade_cls = torch.argmax(pred_invade,dim=1).numpy()
        pred_surgery_cls = torch.argmax(pred_surgery, dim=1).numpy()
        pred_essential_cls = torch.argmax(pred_essential, dim=1).numpy()
        
        for b in range(pred_invade.shape[0]):
            anno_item = anno_items[b]
            if anno_item not in list(pred_invade_dict.keys()):
                pred_invade_dict[anno_item], pred_surgery_dict[anno_item], pred_essential_dict[anno_item] = {}, {}, {}
                score_invade_dict[anno_item], score_surgery_dict[anno_item], score_essential_dict[anno_item] = {}, {}, {}
                label_invade_dict[anno_item], label_surgery_dict[anno_item], label_essential_dict[anno_item] = {}, {}, {}
                feat_dict[anno_item] = {}
            pred_invade_dict[anno_item][img_names[b]]= pred_invade_cls[b]
            pred_surgery_dict[anno_item][img_names[b]] = pred_surgery_cls[b]
            pred_essential_dict[anno_item][img_names[b]] = pred_essential_cls[b]
            score_invade_dict[anno_item][img_names[b]] = pred_invade[b][1].item()
            score_surgery_dict[anno_item][img_names[b]] = pred_surgery[b][1].item()
            score_essential_dict[anno_item][img_names[b]] = pred_essential[b][1].item()
            label_invade_dict[anno_item][img_names[b]] = img_meta['label_invade'][b].item()
            label_surgery_dict[anno_item][img_names[b]] = img_meta['label_surgery'][b].item()
            label_essential_dict[anno_item][img_names[b]] = img_meta['label_essential'][b].item()
            feat_dict[anno_item][img_names[b]] = feat[b]
            
    
    logger.info("Test Complete!!!")
    logger.info("Saving the results...")
    output_path = os.path.join(args.workspace, args.results_test,'results_test.json')
    save_dict = {
        'pred_invade_dict':pred_invade_dict,
        'pred_surgery_dict': pred_surgery_dict,
        "pred_essential_dict": pred_essential_dict,
        'score_invade_dict': score_invade_dict,
        "score_surgery_dict": score_surgery_dict,
        'score_essential_dict': score_essential_dict,
        'label_invade_dict': label_invade_dict,
        'label_surgery_dict': label_surgery_dict,
        'label_essential_dict': label_essential_dict,
        'feat_dict': feat_dict
    }
    save_results(args, save_dict, output_path)
    
    logger.info("============== Calculating Metrics ==============")
    preds, scores, labels, anno_item_list = vote_w_esse(args, pred_invade_dict, pred_surgery_dict, pred_essential_dict, 
                                        score_invade_dict, score_surgery_dict, score_essential_dict,
                                        label_invade_dict, label_surgery_dict, label_essential_dict,
                                        mode=args.score_mode)
    
    metrics_dict = cal_class_metrics(args, logger, preds, scores, labels, args.thres_invade, args.thres_surgery)
    return