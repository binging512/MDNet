import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import ColorJitter
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, confusion_matrix
# from pytorch_grad_cam import GradCAMPlusPlus

from datasets.pan_datasets_3d import Pan_dataset_3d
from models import *
import models.text_encoder.clip as clip
from models.loss import *
from utils.eval_3d import save_results
from utils.post_processing import post_processing
from utils.validate_fasion import slide_window, one_piece
from utils.cal_metrics import cal_class_metrics_3d


def seed_everything():
    seed = 512
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def validate(args, logger, model, dataloader, ep):
    logger.info("============== Validating ==============")
    pred_invade_list, pred_surgery_list, score_invade_list, score_surgery_list = [], [], [], []
    ct_pred_invade_list, ct_pred_surgery_list, ct_score_invade_list, ct_score_surgery_list = [], [], [], []
    anno_item_list = []
    label_invade_list = []
    label_surgery_list = []
    
    for ii, item in enumerate(dataloader):
        if ii%50 == 0:
            logger.info("Validating the {}/{} images...".format(ii,len(dataloader)))
        img, label, img_meta = item
        B,C,D,H,W = img.shape
        anno_item = img_meta['anno_item']
        label_invade = img_meta['label_invade']
        label_surgery = img_meta['label_surgery']
        label_invade = img_meta['gt_invade']
        label_surgery = img_meta['gt_surgery']

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
        
        if args.slide_window:
            output_batch = slide_window(args, model, input_batch)
        else:
            output_batch = one_piece(args, model, input_batch)
            
        for batch_idx in range(B):
            anno_item_list.append(anno_item[batch_idx])
            if args.net_invade_classes == 2:
                score_invade_list.append(output_batch['pred_invade'][batch_idx][1].item())
                score_surgery_list.append(output_batch['pred_surgery'][batch_idx][1].item())
                ct_score_invade_list.append(output_batch['ct_pred_invade'][batch_idx][1].item())
                ct_score_surgery_list.append(output_batch['ct_pred_surgery'][batch_idx][1].item())
            elif args.net_invade_classes>2:
                score_invade_list.append(output_batch['pred_invade'][batch_idx].detach().cpu().numpy().tolist())
                score_surgery_list.append(output_batch['pred_surgery'][batch_idx].detach().cpu().numpy().tolist())
                ct_score_invade_list.append(output_batch['ct_pred_invade'][batch_idx].detach().cpu().numpy().tolist())
                ct_score_surgery_list.append(output_batch['ct_pred_surgery'][batch_idx].detach().cpu().numpy().tolist())
                
            pred_invade_list.append(output_batch['pred_invade_cls'][batch_idx])
            pred_surgery_list.append(output_batch['pred_surgery_cls'][batch_idx])
            ct_pred_invade_list.append(output_batch['ct_pred_invade_cls'][batch_idx])
            ct_pred_surgery_list.append(output_batch['ct_pred_surgery_cls'][batch_idx])
            
    logger.info("Validation Complete!!!")
    logger.info("Saving the results...")
    if dataloader.dataset.mode in ['val']:
        output_path = os.path.join(args.workspace, args.results_val,'results_epoch{}.json'.format(ep))
    elif dataloader.dataset.mode in ['test']:
        output_path = os.path.join(args.workspace, args.results_test,'results_epoch{}.json'.format(ep))
    save_results(args, label_invade_list, label_surgery_list,
                 anno_item_list, score_invade_list, score_surgery_list, pred_invade_list, pred_surgery_list,
                 ct_score_invade_list, ct_score_surgery_list, ct_pred_invade_list, ct_pred_surgery_list, output_path)
    
    logger.info("============== Calculating Metrics ==============")
    logger.info("Classification Results:")

    results = cal_class_metrics_3d(args, logger, 
                         preds=[pred_invade_list,pred_surgery_list],
                         scores=[score_invade_list, score_surgery_list],
                         labels=[label_invade_list, label_surgery_list])
    
    return results['auc_invade'], results['auc_surgery'], results['f1_invade'], results['f1_surgery']


def train_3d(args, logger):
    seed_everything()
    train_dataset = Pan_dataset_3d(args, mode = args.train_mode)
    val_dataset = Pan_dataset_3d(args, mode = 'val')
    test_dataset = Pan_dataset_3d(args, mode = 'test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, drop_last=args.drop_last)
    if args.slide_window:
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
    
    model = Custom_Model_3D(args)
    if os.path.isfile(args.net_resume):
        logger.info("Model training resume from: {}".format(args.net_resume))
        model.load_state_dict(torch.load(args.net_resume), strict=True)
    model = nn.DataParallel(model.cuda())

    # Loss Functions
    if args.net_invade_classes == 2:
        ceLoss_inv = nn.CrossEntropyLoss(ignore_index=255)
    elif args.net_invade_classes == 4:
        weight = torch.tensor([1.,10.,10.,2.]).cuda()
        ceLoss_inv = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
    ceLoss_sur = nn.CrossEntropyLoss(ignore_index=255)
    
    bceLoss_inv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5, 2]).cuda())
    bceLoss_sur = nn.BCEWithLogitsLoss()
    # distsoftloss = DistsoftBCELoss(dist_scale=5)
    distsoftloss = DistsoftCELoss(dist_scale=5)
    seg_ceLoss = nn.CrossEntropyLoss(weight=torch.tensor([1.,10.,10.]).cuda(), ignore_index=255)
    diceLoss = DiceLoss(ignore_index=255)
    
    if args.net_name.lower() in ['swin']:
        optimizer = AdamW(model.parameters(),lr=args.net_learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
        # optimizer = AdamW([{'params':list(model.module.img_params.parameters()), 'lr':args.net_learning_rate/10},
        #                     {'params':list(model.module.text_params.parameters()), 'lr':args.net_learning_rate/10},
        #                     {'params':list(model.module.class_params.parameters()), 'lr':args.net_learning_rate},])
        scheduler = SequentialLR(optimizer, 
                             schedulers=[
                                 LinearLR(optimizer, start_factor=0.5, total_iters=int(args.net_num_epoches*len(train_dataloader)*0.1)),
                                 CosineAnnealingLR(optimizer,int(args.net_num_epoches*len(train_dataloader)*0.9), eta_min=1e-8)
                             ],
                             milestones=[int(args.net_num_epoches*len(train_dataloader)*0.1)])
    else:
        optimizer = AdamW(model.parameters(),lr=args.net_learning_rate)
        scheduler = SequentialLR(optimizer, 
                             schedulers=[
                                 LinearLR(optimizer, start_factor=0.5, total_iters=int(args.net_num_epoches*len(train_dataloader)*0.1)),
                                 CosineAnnealingLR(optimizer,int(args.net_num_epoches*len(train_dataloader)*0.9), eta_min=1e-8)
                             ],
                             milestones=[int(args.net_num_epoches*len(train_dataloader)*0.1)])
        # scheduler = CosineAnnealingLR(optimizer, T_max=args.net_num_epoches*len(train_dataloader), eta_min=1e-7)
    
    best_metric = 0
    best_invade_auc = 0
    best_surgery_auc = 0
    best_invade_f1 = 0
    best_surgery_f1 = 0
    best_epoch = 0
    best_mode = 'None'
    try:
        main_part, main_metric = args.main_metric.split('_')
    except:
        main_part, main_metric = 'test', 'inv'
        
    for ep in range(args.net_num_epoches):
        logger.info("==============Training {}/{} Epoches==============".format(ep,args.net_num_epoches))
        for ii, item in enumerate(train_dataloader):
            optimizer.zero_grad()
            img, label, img_meta = item
            img = img.cuda()
            label = label.squeeze(1).cuda()
            label_invade = img_meta['label_invade'].cuda()
            label_surgery = img_meta['label_surgery'].cuda()
            input_batch = {
                "img": img,
                "blood": img_meta['blood'].cuda(),
                'others': img_meta['others'].cuda(),
                "blood_des": clip.tokenize(img_meta['blood_des'], context_length=256).cuda(),
                "others_des": clip.tokenize(img_meta['others_des']).cuda(),
            }
            
            output = model(input_batch)
            
            loss = 0
            loss_dict = {}
            # Calculate the losses
            for iii in range(len(output['preds_invade'])):
                if args.net_invade_cls_celoss:
                    if args.D_soft_sample:
                        loss_bce_invade = bceLoss_inv(output['preds_invade'][iii], label_invade)
                        loss += loss_bce_invade
                        loss_dict['loss_bce_invade'] = loss_bce_invade
                    else:
                        loss_ce_invade = ceLoss_inv(output['preds_invade'][iii], label_invade.long()) 
                        loss += loss_ce_invade
                        loss_dict["loss_ce_invade"] = loss_ce_invade
                if args.net_surgery_cls_celoss:
                    if args.D_soft_sample:
                        loss_bce_surgery = bceLoss_sur(output['preds_surgery'][iii], label_surgery)
                        loss += loss_bce_surgery
                        loss_dict['loss_bce_surgery'] = loss_bce_surgery
                    else:
                        loss_ce_surgery = ceLoss_sur(output['preds_surgery'][iii], label_surgery.long())
                        loss += loss_ce_surgery
                        loss_dict['loss_ce_surgery'] = loss_ce_surgery

            for iii in range(len(output['ct_preds_invade'])):
                if args.net_ct_invade_cls_celoss:
                    loss_ce_ct_invade = ceLoss_inv(output['ct_preds_invade'][iii], label_invade.long())*0.1
                    loss += loss_ce_ct_invade
                    loss_dict["loss_ce_ct_invade"] = loss_ce_ct_invade
                if args.net_ct_surgery_cls_celoss:
                    loss_ce_ct_surgery = ceLoss_sur(output['ct_preds_surgery'][iii], label_surgery.long())*0.1
                    loss += loss_ce_ct_surgery
                    loss_dict["loss_ce_ct_surgery"] = loss_ce_ct_surgery
                
            if args.net_seg_celoss and output['pred_seg'] != None:
                loss_ce_seg = seg_ceLoss(output['pred_seg'], label.long())*0.1
                loss += loss_ce_seg
                loss_dict['loss_ce_seg'] = loss_ce_seg
            if args.net_seg_diceloss and output['pred_seg'] != None:
                loss_dice_seg = diceLoss(output['pred_seg'], label.long())*0.1
                loss += loss_dice_seg
                loss_dict['loss_dice_seg'] = loss_dice_seg
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if ii%5 == 0:
                logger.info("Epoch:{}/{} || iter:{}/{} || loss:{:.4f} || lr:{}".format(ep, args.net_num_epoches,
                                                                                    ii, len(train_dataloader),
                                                                                    loss, scheduler.get_last_lr()[0]))
                logger.info("Loss details:{}".format(["{}:{:.8f}".format(k,v) for k,v in loss_dict.items()]))

        model.eval()
        if (ep+1)%args.val_interval == 0:
            if args.train_mode in ['train']:
                logger.info("=================Evaluation for Val Dataset=================")
                val_auc_invade, val_auc_surgery, val_f1_invade, val_f1_surgery = validate(args, logger, model, val_dataloader, ep)
            logger.info("=================Evaluation for Test Dataset=================")
            test_auc_invade, test_auc_surgery, test_f1_invade, test_f1_surgery = validate(args, logger, model, test_dataloader, ep)
                
            if main_part.lower() in ['val'] and args.train_mode in ['train']:
                auc_invade = val_auc_invade
                auc_surgery = val_auc_surgery
                f1_invade = val_f1_invade
                f1_surgery = val_f1_surgery
                pass
            elif main_part.lower() in ['test']:
                auc_invade = test_auc_invade
                auc_surgery = test_auc_surgery
                f1_invade = test_f1_invade
                f1_surgery = test_f1_surgery
            else:
                raise NotImplementedError('Selected part {} error!'.format(main_part))
                
            if main_metric.lower() in ['inv','invade']:
                select_metric = auc_invade
            elif main_metric.lower() in ['sur','surgery']:
                select_metric = auc_surgery
            elif main_metric.lower() in ['avg', 'average']:
                select_metric = (auc_invade+auc_surgery)/2
            else:
                raise NotImplementedError('Selected metric {} error!'.format(main_metric))
            
            if select_metric > best_metric:
                best_metric = select_metric
                best_invade_auc = auc_invade
                best_surgery_auc = auc_surgery
                best_invade_f1 = f1_invade
                best_surgery_f1 = f1_surgery
                best_epoch = ep
                best_mode = 'Classification'
                torch.save(model.module.state_dict(), os.path.join(args.workspace,'best_model.pth'))
                logger.info("Best mode: {}.".format(best_mode))
                logger.info("Best model update: Epoch:{}, AUC Invade:{:.5f}, AUC Surgery:{:.5f}, F1 Invade:{:.5f}, F1 Surgery:{:.5f}".format(best_epoch, 
                                                                                                                                            best_invade_auc, 
                                                                                                                                            best_surgery_auc, 
                                                                                                                                            best_invade_f1,
                                                                                                                                            best_surgery_f1))
                logger.info("Best model saved to: {}".format(os.path.join(args.workspace,'best_model.pth')))
            logger.info("Select part: {}. Select metric: {}, {:.5f}.".format(main_part, main_metric, select_metric))
            
        if (ep+1)%args.save_interval==0:
            torch.save(model.module.state_dict(), os.path.join(args.workspace,'epoch_{}.pth'.format(ep+1)))
            logger.info("Checkpoint saved to {}".format(os.path.join(args.workspace,'epoch_{}.pth'.format(ep+1))))
        
        logger.info("Best mode:{}, Best Epoch:{}, Best AUC Invade:{:.5f}, Best AUC Surgery:{:.5f}, Best F1 Invade:{:.5f}, Best F1 Surgery:{:.5f}".format(best_mode,
                                                                                                                                                        best_epoch, 
                                                                                                                                                        best_invade_auc, 
                                                                                                                                                        best_surgery_auc, 
                                                                                                                                                        best_invade_f1,
                                                                                                                                                        best_surgery_f1))
        model.train()
    torch.save(model.module.state_dict(), os.path.join(args.workspace,'final.pth'))
