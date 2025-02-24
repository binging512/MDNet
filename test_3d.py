import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import ColorJitter
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from datasets.pan_datasets_3d import Pan_dataset_3d
from models import *
import models.text_encoder.clip as clip
from models.loss import *
from utils.eval_3d import save_results_test
from utils.cal_metrics import cal_class_metrics_3d

def test_3d(args, logger):
    test_dataset = Pan_dataset_3d(args, mode = 'test')
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    model = Custom_Model_3D(args).cuda()
    
    model.load_state_dict(torch.load(args.test_checkpoint))
    model.eval()
    
    logger.info("============== Testing ==============")
    pred_invade_list, pred_surgery_list, score_invade_list, score_surgery_list = [], [], [], []
    ct_pred_invade_list, ct_pred_surgery_list, ct_score_invade_list, ct_score_surgery_list = [], [], [], []
    anno_item_list = []
    label_invade_list = []
    label_surgery_list = []
    img_feat_list = []
    
    for ii, item in enumerate(dataloader):
        if ii%50 == 0:
            logger.info("Testing the {}/{} images...".format(ii,len(dataloader)))
        img, label, img_meta = item
        B,C,D,H,W = img.shape
        anno_item = img_meta['anno_item']
        label_invade = img_meta['label_invade']
        label_surgery = img_meta['label_surgery']

        for l_invade in label_invade:
            label_invade_list.append(l_invade.item())
        for l_surgery in label_surgery:
            label_surgery_list.append(l_surgery.item())
            
        input_batch = {
            "img": img.cuda(),
            "blood": img_meta['blood'].cuda(),
            'others': img_meta['others'].cuda(),
            "blood_des": clip.tokenize(img_meta['blood_des'], context_length=256).cuda(),
            "others_des": clip.tokenize(img_meta['others_des']).cuda(),
        }
        
        with torch.no_grad():
            output = model(input_batch)
            
        preds_invade = output['preds_invade']
        preds_surgery = output['preds_surgery']
        ct_preds_invade = output['ct_preds_invade']
        ct_preds_surgery = output['ct_preds_surgery']
        img_feature = output['img_feature']
                
        # Fusion
        if args.test_fusion =='mean':
            pred_invade = torch.mean(torch.stack(preds_invade,dim=0), dim=0)
            pred_surgery = torch.mean(torch.stack(preds_surgery,dim=0), dim=0)
            ct_pred_invade = torch.mean(torch.stack(ct_preds_invade,dim=0), dim=0)
            ct_pred_surgery = torch.mean(torch.stack(ct_preds_surgery,dim=0), dim=0)
        if args.test_fusion == 'max':
            pred_invade,_ = torch.max(torch.stack(preds_invade,dim=0), dim=0)
            pred_surgery,_ = torch.max(torch.stack(preds_surgery,dim=0), dim=0)
            ct_pred_invade,_ = torch.max(torch.stack(ct_preds_invade,dim=0), dim=0)
            ct_pred_surgery,_ = torch.max(torch.stack(ct_preds_surgery,dim=0), dim=0)
            
        pred_invade = torch.softmax(pred_invade, dim=1)
        pred_surgery = torch.softmax(pred_surgery, dim=1)
        ct_pred_invade = torch.softmax(ct_pred_invade, dim=1)
        ct_pred_surgery = torch.softmax(ct_pred_surgery, dim=1)
        
        pred_invade_cls = torch.argmax(pred_invade.detach().cpu(),dim=1).numpy()
        pred_surgery_cls = torch.argmax(pred_surgery.detach().cpu(),dim=1).numpy()
        ct_pred_invade_cls = torch.argmax(ct_pred_invade.detach().cpu(),dim=1).numpy()
        ct_pred_surgery_cls = torch.argmax(ct_pred_surgery.detach().cpu(),dim=1).numpy()
        
        for batch_idx in range(B):
            anno_item_list.append(anno_item[batch_idx])
            score_invade_list.append(torch.sum(pred_invade[batch_idx][1:]).item())
            score_surgery_list.append(pred_surgery[batch_idx][1].item())
            pred_invade_list.append(pred_invade_cls[batch_idx])
            pred_surgery_list.append(pred_surgery_cls[batch_idx])
            ct_score_invade_list.append(torch.sum(ct_pred_invade[batch_idx][1:]).item())
            ct_score_surgery_list.append(ct_pred_surgery[batch_idx][1].item())
            ct_pred_invade_list.append(ct_pred_invade_cls[batch_idx])
            ct_pred_surgery_list.append(ct_pred_surgery_cls[batch_idx])
            img_feat_list.append(img_feature[batch_idx].detach().cpu().numpy().tolist())
            
    logger.info("Validation Complete!!!")
    logger.info("Saving the results...")
    output_path = os.path.join(args.workspace, args.results_test,'results_test.json')
    save_results_test(args, label_invade_list, label_surgery_list,
                        anno_item_list, score_invade_list, score_surgery_list, pred_invade_list, pred_surgery_list,
                        ct_score_invade_list, ct_score_surgery_list, ct_pred_invade_list, ct_pred_surgery_list, 
                        img_feat_list, output_path)
      
    logger.info("============== Calculating Metrics ==============")
    results = cal_class_metrics_3d(args, logger, 
                                    preds=[pred_invade_list, pred_surgery_list],
                                    scores=[score_invade_list, score_surgery_list],
                                    labels=[label_invade_list, label_surgery_list])
    return