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
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, accuracy_score
from sklearn.preprocessing import label_binarize

from datasets.pan_datasets_2d import Pan_dataset_2d, get_class_weighted_indice, EqualSampler

from models import Custom_Model_2D
from models.img_encoder_2d.UNet2d import UNet
import models.text_encoder.clip as clip
from models.loss import *

from utils.eval_2d import vote_w_esse, save_results, vote_w_esse_v2, eval_metrics, save_results_post
from utils.cal_metrics import cal_class_metrics

from IPython.core.debugger import set_trace

def seed_everything():
    seed = 512
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def validate(args, logger, model, dataloader, cur_iter, thres_invade=0.5, thres_surgery=0.5):
    logger.info("============== Validating ==============")
    pred_invade_dict,pred_surgery_dict, pred_essential_dict= {},{},{}
    score_invade_dict, score_surgery_dict, score_essential_dict = {},{},{}
    label_invade_dict, label_surgery_dict, label_essential_dict = {},{},{}
    feat_dict = {}
    IoU_list, Dice_list = [], []
    
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
            pred_invade_dict[anno_item][img_names[b]]= pred_invade_cls[b]
            pred_surgery_dict[anno_item][img_names[b]] = pred_surgery_cls[b]
            pred_essential_dict[anno_item][img_names[b]] = pred_essential_cls[b]
            score_invade_dict[anno_item][img_names[b]] = pred_invade[b][1].item()
            score_surgery_dict[anno_item][img_names[b]] = pred_surgery[b][1].item()
            score_essential_dict[anno_item][img_names[b]] = pred_essential[b][1].item()
            label_invade_dict[anno_item][img_names[b]] = img_meta['label_invade'][b].item()
            label_surgery_dict[anno_item][img_names[b]] = img_meta['label_surgery'][b].item()
            label_essential_dict[anno_item][img_names[b]] = img_meta['label_essential'][b].item()
        # Segmentation
        # preds_seg = torch.argmax(preds_seg, dim=1).detach().cpu().numpy()
        # gts_seg = label.detach().cpu().numpy()
        # for b in range(preds_seg.shape[0]):
        #     ret_metrics = eval_metrics(preds_seg[b], gts_seg[b], num_classes=args.net_seg_classes, ignore_index=255, metrics=['mIoU', 'mDice'])
        #     IoU_list.append(ret_metrics['IoU'])
        #     Dice_list.append(ret_metrics['Dice'])

    logger.info("Validation Complete!!!")
    logger.info("Saving the results...")
    if dataloader.dataset.mode in ['val']:
        output_path = os.path.join(args.workspace, args.results_val,'results_iter{}.json'.format(cur_iter))
    elif dataloader.dataset.mode in ['test']:
        output_path = os.path.join(args.workspace, args.results_test,'results_iter{}.json'.format(cur_iter))
    elif dataloader.dataset.mode in ['val_test']:
        output_path = os.path.join(args.workspace, args.results_test,'results_val_test_iter{}.json'.format(cur_iter))
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
    }
    save_results(args, save_dict, output_path.replace('.json','_score.json'))
    
    logger.info("============== Calculating Metrics ==============")
    preds, scores, labels, anno_item_list = vote_w_esse(args, pred_invade_dict, pred_surgery_dict, pred_essential_dict, 
                                        score_invade_dict, score_surgery_dict, score_essential_dict,
                                        label_invade_dict, label_surgery_dict, label_essential_dict,
                                        mode=args.score_mode)
    
    save_results_post(args, output_path, anno_item_list, preds, scores, labels)
    
    metrics_dict = cal_class_metrics(args, logger, preds, scores, labels, thres_invade, thres_surgery)
    # Segmentation
    # IoU = np.nanmean(np.stack(IoU_list, axis=0), axis=0)
    # Dice = np.nanmean(np.stack(Dice_list, axis=0), axis=0)
    # logger.info("mIoU: {:.5f}".format(np.mean(IoU)))
    # logger.info("IoU-bg: {:.5f}, IoU-tumor: {:.5f}, IoU-vein: {:.5f}".format(IoU[0],IoU[1],IoU[2]))
    # logger.info("mDice: {:.5f}".format(np.mean(Dice)))
    # logger.info("Dice-bg: {:.5f}, Dice-tumor: {:.5f}, Dice-vein: {:.5f}".format(Dice[0],Dice[1],Dice[2]))
    
    auc_invade, auc_surgery, auc_essential = metrics_dict["auc_invade"], metrics_dict["auc_surgery"], metrics_dict["auc_essential"]
    f1_invade, f1_surgery, f1_essential = metrics_dict['f1_invade'], metrics_dict['f1_surgery'], metrics_dict['f1_essential']
    best_thres_invade, best_thres_surgery = metrics_dict["thres_invade"], metrics_dict["thres_surgery"]
    return auc_invade, auc_surgery, auc_essential, f1_invade, f1_surgery, f1_essential, best_thres_invade, best_thres_surgery

def train_2d(args, logger):
    seed_everything()
    train_dataset = Pan_dataset_2d(args, mode = args.train_mode)
    val_dataset = Pan_dataset_2d(args, mode = 'val')
    test_dataset = Pan_dataset_2d(args, mode = 'test')
    
    sampler = EqualSampler(train_dataset, batch_size=args.batch_size)
    # sampler = SubsetRandomSampler(indices=torch.randperm(len(train_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=sampler, drop_last=args.drop_last)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
    
    
    model = Custom_Model_2D(args)
    
    if os.path.isfile(args.net_resume):
        logger.info("Model training resume from: {}".format(args.net_resume))
        model.load_state_dict(torch.load(args.net_resume), strict=True)
    model = nn.DataParallel(model.cuda())

    if args.net_invade_classes == 4:
        weights = torch.tensor(args.net_invade_weights).cuda()
        inv_cls_ceLoss = Cls_CELoss(args, weight=weights)
    else:
        inv_cls_ceLoss = Cls_CELoss(args, weight=None)
    
    sur_cls_ceLoss = Cls_CELoss(args, weight=None)
    esse_cls_ceLoss = Cls_CELoss(args, weight=None)
    seg_ceLoss = Seg_CELoss(args, weights=torch.tensor(args.net_seg_weights).cuda())
    seg_diceLoss = Seg_DiceLoss(args, weights=torch.tensor(args.net_seg_weights).cuda())
    
    # optimizer = AdamW(model.parameters(),lr=args.net_learning_rate)
    # optimizer = SGD([{'params':list(model.module.pretrained.parameters()), 'lr':args.net_learning_rate/10},
    #                         {'params':list(model.module.new_added.parameters()), 'lr':args.net_learning_rate}])
    # optimizer = Adam([{'params':list(model.module.pretrained.parameters()), 'lr':args.net_learning_rate/10},
    #                         {'params':list(model.module.new_added.parameters()), 'lr':args.net_learning_rate}])
    if args.net_name.lower() in ['resnet','cvt']:
        optimizer = AdamW([{'params':list(model.module.pretrained.parameters()), 'lr':args.net_learning_rate/10},
                            {'params':list(model.module.new_added.parameters()), 'lr':args.net_learning_rate}])
    elif args.net_name in ['swin','vit']:
        optimizer = AdamW([{'params':list(model.module.pretrained.parameters()), 'lr':args.net_learning_rate/100},
                            {'params':list(model.module.new_added.parameters()), 'lr':args.net_learning_rate}])
    else:
        raise NotImplementedError("Optimizer for Model {} is not implemented!".format(args.net_name))
    
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.net_num_iter, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, 
                             schedulers=[
                                 LinearLR(optimizer, start_factor=0.1, total_iters=int(args.net_num_iter*0.1)),
                                 CosineAnnealingLR(optimizer,int(args.net_num_iter*0.9), eta_min=1e-6)
                             ],
                             milestones=[int(args.net_num_iter*0.1)])

    best_metric = 0 
    best_invade_auc, best_surgery_auc, best_essential_auc = 0, 0, 0
    best_invade_f1, best_surgery_f1, best_essential_f1 = 0, 0, 0
    best_iter = 0
    best_mode = 'None'
    
    try:
        main_part, main_metric = args.main_metric.split('_')
    except:
        main_part, main_metric = 'test', 'inv'
    
    flags = 0
    cur_iter = 0
    total_epoch = int(args.net_num_iter/len(train_dataloader))+1
    # Stage 1
    for ep in range(total_epoch):
        print("Length of dataloader:{}".format(len(train_dataloader)))
        logger.info("==============Training {}/{} Epoches==============".format(ep,total_epoch))
        for ii, item in enumerate(train_dataloader):
            if (cur_iter+1)%args.val_interval == 0:
                logger.info("==============Training {}/{} Iterations==============".format(cur_iter, args.net_num_iter))
            optimizer.zero_grad()
            img, label, img_meta = item
            img = img.cuda()
            label = label.cuda()
            label_invade = img_meta['label_invade'].cuda()
            label_surgery = img_meta['label_surgery'].cuda()
            label_essential = img_meta['label_essential'].cuda()
            
            input_batch = {
                'img': img,
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
                    loss_ce_invade = inv_cls_ceLoss(output['preds_invade'][iii], label_invade.long(), label_essential)
                    loss += loss_ce_invade
                    loss_dict["loss_ce_invade"] = loss_ce_invade
                if args.net_surgery_cls_celoss:
                    loss_ce_surgery = sur_cls_ceLoss(output['preds_surgery'][iii], label_surgery.long(), label_essential)
                    loss += loss_ce_surgery
                    loss_dict["loss_ce_surgery"] = loss_ce_surgery
                if args.net_essential_cls_celoss:
                    loss_ce_essential = esse_cls_ceLoss(output['preds_essential'][iii], label_essential.long())
                    loss += loss_ce_essential
                    loss_dict["loss_ce_essential"] = loss_ce_essential
                
            if args.net_seg_celoss:
                loss_ce_seg = seg_ceLoss(output['pred_seg'], label.long(), label_essential.long())
                loss += loss_ce_seg
                loss_dict["loss_ce_seg"] = loss_ce_seg
                    
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if cur_iter%10 == 0:
                logger.info("Epoch:{}/{} || iter:{}/{} || loss:{:.4f} || lr:{}".format(ep, total_epoch,
                                                                                    cur_iter, args.net_num_iter,
                                                                                    loss, scheduler.get_last_lr()[0]))
                logger.info("Loss details:{}".format(["{}:{:.8f}".format(k,v) for k,v in loss_dict.items()]))
                
            if (cur_iter+1)%args.val_interval == 0:
                model.eval()
                logger.info("=================Evaluation for Val Dataset=================")
                val_auc_invade, val_auc_surgery, val_auc_essential, val_f1_invade, val_f1_surgery, val_f1_essential, best_thres_invade, best_thres_surgery = validate(args, logger, model, val_dataloader, cur_iter)
                logger.info("=================Evaluation for Test Dataset=================")
                test_auc_invade, test_auc_surgery, test_auc_essential, test_f1_invade, test_f1_surgery, test_f1_essential, _, _ = validate(args, logger, model, test_dataloader, cur_iter, best_thres_invade, best_thres_surgery)
                
                if main_part.lower() in ['val'] and args.train_mode in ['train']:
                    auc_invade, auc_surgery, auc_essential = val_auc_invade, val_auc_surgery, val_auc_essential
                    f1_invade, f1_surgery,f1_essential= val_f1_invade, val_f1_surgery, val_f1_essential
                elif main_part.lower() in ['test']:
                    auc_invade, auc_surgery, auc_essential = test_auc_invade, test_auc_surgery, test_auc_essential
                    f1_invade, f1_surgery,f1_essential= test_f1_invade, test_f1_surgery, test_f1_essential
                else:
                    raise NotImplementedError('Selected part {} error!'.format(main_part))
                
                if main_metric.lower() in ['inv','invade']:
                    select_metric = auc_invade
                elif main_metric.lower() in ['sur','surgery']:
                    select_metric = auc_surgery
                elif main_metric.lower() in ['esse','essential']:
                    select_metric = auc_essential
                elif main_metric.lower() in ['avg', 'average']:
                    select_metric = (auc_invade+auc_surgery)/2
                else:
                    raise NotImplementedError('Selected metric {} error!'.format(main_metric))
                
                logger.info('======Multi-modality======')
                logger.info('Select metric: {:.5f}'.format(select_metric))
                logger.info("Iter:{}, AUC Invade:{:.5f}, AUC Surgery:{:.5f}, AUC Essential:{:.5f}, F1 Invade:{:.5f}, F1 Surgery:{:.5f}, F1 Essential:{:.5f}".format( 
                                cur_iter, auc_invade, auc_surgery, auc_essential, f1_invade, f1_surgery, f1_essential))
                
                if select_metric > best_metric:
                    best_metric = select_metric
                    best_invade_auc = auc_invade
                    best_surgery_auc = auc_surgery
                    best_essential_auc = auc_essential
                    best_invade_f1 = f1_invade
                    best_surgery_f1 = f1_surgery
                    best_essential_f1 = f1_essential
                    best_iter = cur_iter
                    best_mode = 'Classification'
                    torch.save(model.module.state_dict(), os.path.join(args.workspace,'best_model.pth'))
                    logger.info("Best mode: {}".format(best_mode))
                    logger.info("Best model update: Iter:{}, AUC Invade:{:.5f}, AUC Surgery:{:.5f}, AUC Essential:{:.5f}, F1 Invade:{:.5f}, F1 Surgery:{:.5f}, F1 Essential:{:.5f}".format(best_iter, 
                                                                                                                                                best_invade_auc, 
                                                                                                                                                best_surgery_auc,
                                                                                                                                                best_essential_auc, 
                                                                                                                                                best_invade_f1,
                                                                                                                                                best_surgery_f1,
                                                                                                                                                best_essential_f1))
                    logger.info("Best model saved to: {}".format(os.path.join(args.workspace,'best_model.pth')))
                model.train()
                
                if auc_essential>=args.stage_thres and args.two_stage:
                    flags = 1
                    cur_iter +=1
                    logger.info("Dest AUC Essential Qualified: {:.5f}".format(auc_essential))
                    torch.save(model.module.state_dict(), os.path.join(args.workspace,'stage1_iter_{}.pth'.format(ii+1)))
                    logger.info("Checkpoint saved to {}".format(os.path.join(args.workspace,'stage1_iter_{}.pth'.format(ii+1))))
                    break
            
            if (cur_iter+1)%args.save_interval==0:
                os.makedirs(os.path.join(args.workspace,'checkpoints'),exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.workspace, 'checkpoints', 'iter_{}.pth'.format(cur_iter+1)))
                logger.info("Checkpoint saved to {}".format(os.path.join(args.workspace, 'checkpoints', 'iter_{}.pth'.format(cur_iter+1))))
            
            cur_iter +=1
            if cur_iter>=args.net_num_iter or flags==1:
                break
            
    logger.info("Best mode:{}, Best Iter:{}, Best AUC Invade:{:.5f}, Best AUC Surgery:{:.5f}, Best AUC Essential:{:.5f}, Best F1 Invade:{:.5f}, Best F1 Surgery:{:.5f}, Best F1 Essential:{:.5f}".format(best_mode,
                                                                                                                            best_iter, 
                                                                                                                            best_invade_auc, 
                                                                                                                            best_surgery_auc, 
                                                                                                                            best_essential_auc,
                                                                                                                            best_invade_f1,
                                                                                                                            best_surgery_f1,
                                                                                                                            best_essential_f1))
            
    torch.save(model.module.state_dict(), os.path.join(args.workspace,'final.pth'))
