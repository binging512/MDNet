import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from sklearn.preprocessing import label_binarize
from collections import OrderedDict
import mmcv
import os
import cv2

def mean_topk(score_list, topk):
    if isinstance(score_list, np.ndarray):
        score_list = score_list.tolist()
    score_list =sorted(score_list, reverse=True)
    mean_score = np.mean(score_list[:topk])
    return mean_score

def vote_w_esse(args, pred_invade_dict, pred_surgery_dict, pred_essential_dict, 
                score_invade_dict, score_surgery_dict, score_essential_dict,
                label_invade_dict, label_surgery_dict, label_essential_dict,
                mode = 'avg'):
    vote_mode = args.vote_mode
    preds_invade, preds_surgery, preds_essential = [], [], []
    scores_invade, scores_surgery, scores_essential = [], [], []
    labels_invade, labels_surgery, labels_essential = [], [], []
    
    for anno_item in list(pred_invade_dict.keys()):
        pred_invade_list = np.array([v for k,v in pred_invade_dict[anno_item].items()])
        pred_surgery_list = np.array([v for k,v in pred_surgery_dict[anno_item].items()])
        pred_essential_list = np.array([v for k,v in pred_essential_dict[anno_item].items()])
        score_invade_list = np.array([v for k,v in score_invade_dict[anno_item].items()])
        score_surgery_list = np.array([v for k,v in score_surgery_dict[anno_item].items()])
        score_essential_list = np.array([v for k,v in score_essential_dict[anno_item].items()])
        label_invade_list = np.array([v for k,v in label_invade_dict[anno_item].items()])
        label_surgery_list = np.array([v for k,v in label_surgery_dict[anno_item].items()])
        label_essential_list = np.array([v for k,v in label_essential_dict[anno_item].items()])
        if vote_mode.lower() in ['none']:
            if mode.lower() in ['avg']:
                pred_invade = 1 if np.mean(pred_invade_list)>=0.5 else 0
                pred_surgery = 1 if np.mean(pred_surgery_list)>=0.5 else 0
                score_invade = np.mean(pred_invade_list)
                score_surgery = np.mean(score_surgery_list)
            elif mode.lower() in ['max']:
                pred_invade = 1 if np.max(pred_invade_list)>=0.5 else 0
                pred_surgery = 1 if np.max(pred_surgery_list)>=0.5 else 0
                score_invade = np.max(pred_invade_list)
                score_surgery = np.max(score_surgery_list)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        elif vote_mode.lower() in ['pred','pred_essential']:
            if mode.lower() in ['avg']:
                pred_invade = 1 if np.sum(pred_invade_list*pred_essential_list)/(np.sum(pred_essential_list)+1e-5)>=0.5 else 0
                pred_surgery = 1 if np.sum(pred_surgery_list*pred_essential_list)/(np.sum(pred_essential_list)+1e-5)>=0.5 else 0
                score_invade = np.sum(score_invade_list*pred_essential_list)/(np.sum(pred_essential_list)+1e-5)
                score_surgery = np.sum(score_surgery_list*pred_essential_list)/(np.sum(pred_essential_list)+1e-5)
            elif mode.lower() in ['max']:
                pred_invade = 1 if np.max((np.array(pred_invade_list)>=0.5)*(np.array(pred_essential_list)>=0.5)*1) else 0
                pred_surgery = 1 if np.max((np.array(pred_surgery_list)>=0.5)*(np.array(pred_essential_list)>=0.5)*1) else 0
                score_invade = np.max(score_invade_list*pred_essential_list)/(np.max(pred_essential_list)+1e-5)
                score_surgery = np.max(score_surgery_list*pred_essential_list)/(np.max(pred_essential_list)+1e-5)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        elif vote_mode.lower() in ['label', 'label_essential']:
            if mode.lower() in ['avg']:
                pred_invade = 1 if np.sum(pred_invade_list*label_essential_list)/(np.sum(label_essential_list)+1e-5)>=0.5 else 0
                pred_surgery = 1 if np.sum(pred_surgery_list*label_essential_list)/(np.sum(label_essential_list)+1e-5)>=0.5 else 0
                score_invade = np.sum(score_invade_list*label_essential_list)/(np.sum(label_essential_list)+1e-5)
                score_surgery = np.sum(score_surgery_list*label_essential_list)/(np.sum(label_essential_list)+1e-5)
            elif mode.lower() in ['max']:
                pred_invade = 1 if np.max((np.array(pred_invade_list)>=0.5)*(np.array(label_essential_list)>=0.5)*1) else 0
                pred_surgery = 1 if np.max((np.array(pred_surgery_list)>=0.5)*(np.array(label_essential_list)>=0.5)*1) else 0
                score_invade = np.max(score_invade_list*label_essential_list)/(np.max(label_essential_list)+1e-5)
                score_surgery = np.max(score_surgery_list*label_essential_list)/(np.max(label_essential_list)+1e-5)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        elif vote_mode.lower() in ['score','score_essential']:
            if mode.lower() in ['avg']:
                pred_invade = 1 if np.sum(pred_invade_list*score_essential_list)/(np.sum(score_essential_list)+1e-5)>=0.5 else 0
                pred_surgery = 1 if np.sum(pred_surgery_list*score_essential_list)/(np.sum(score_essential_list)+1e-5)>=0.5 else 0
                score_invade = np.sum(score_invade_list*score_essential_list)/(np.sum(score_essential_list)+1e-5)
                score_surgery = np.sum(score_surgery_list*score_essential_list)/(np.sum(score_essential_list)+1e-5)
            elif mode.lower() in ['max']:
                pred_invade = 1 if mean_topk(score_invade_list*score_essential_list, 10)>=0.5 else 0
                pred_surgery = 1 if mean_topk(score_surgery_list*score_essential_list, 10)>=0.5 else 0
                score_invade = mean_topk(score_invade_list*score_essential_list, 10)
                score_surgery = mean_topk(score_surgery_list*score_essential_list, 10)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        else:
            raise NotImplementedError("Voting mode: {} is not implemented!".format(vote_mode))
        label_invade = label_invade_list[0]
        label_surgery = label_surgery_list[0]
        
        preds_invade.append(pred_invade)
        preds_surgery.append(pred_surgery)
        preds_essential += pred_essential_list.tolist()
        scores_invade.append(score_invade)
        scores_surgery.append(score_surgery)
        scores_essential += score_essential_list.tolist()
        labels_invade.append(label_invade)
        labels_surgery.append(label_surgery)
        labels_essential += label_essential_list.tolist()
        
    anno_item_list = [anno_item for anno_item in list(pred_invade_dict.keys())]
        
    return [preds_invade, preds_surgery, preds_essential],\
            [scores_invade, scores_surgery, scores_essential],\
            [labels_invade, labels_surgery, labels_essential],\
            anno_item_list

def vote_w_esse_v2(args, pred_invade_dict, pred_surgery_dict, pred_essential_dict, 
                score_invade_dict, score_surgery_dict, score_essential_dict,
                label_invade_dict, label_surgery_dict, label_essential_dict,
                mode = 'avg'):
    vote_mode = args.vote_mode
    preds_invade, preds_surgery, preds_essential = [], [], []
    scores_invade, scores_surgery, scores_essential = [], [], []
    labels_invade, labels_surgery, labels_essential = [], [], []
    for anno_item in list(pred_invade_dict.keys()):
        pred_invade_list = label_binarize(np.array([v for k,v in pred_invade_dict[anno_item].items()]), classes=[i for i in range(args.net_invade_classes)])
        pred_surgery_list = label_binarize(np.array([v for k,v in pred_surgery_dict[anno_item].items()]), classes=[i for i in range(args.net_surgery_classes)])
        pred_essential_list = np.array([v for k,v in pred_essential_dict[anno_item].items()])[:,np.newaxis]
        score_invade_list = [v.numpy() for k,v in score_invade_dict[anno_item].items()]
        score_surgery_list = [v.numpy() for k,v in score_surgery_dict[anno_item].items()]
        score_essential_list = np.array([v for k,v in score_essential_dict[anno_item].items()])[:,np.newaxis]
        label_invade_list = np.array([v for k,v in label_invade_dict[anno_item].items()])
        label_surgery_list = np.array([v for k,v in label_surgery_dict[anno_item].items()])
        label_essential_list = np.array([v for k,v in label_essential_dict[anno_item].items()])[:,np.newaxis]
        
        if vote_mode.lower() in ['none']:
            if mode.lower() in ['avg']:
                pred_invade = np.argmax(np.sum(pred_invade_list, axis=0))
                pred_surgery = np.argmax(np.sum(pred_surgery_list, axis=0))
                score_invade = np.mean(score_invade_list, axis=0)
                score_surgery = np.mean(score_surgery_list, axis=0)
            elif mode.lower() in ['max']:
                pred_invade = np.max(np.where(np.sum(pred_invade_list, axis=0)))
                pred_surgery = np.max(np.where(np.sum(pred_surgery_list, axis=0)))
                score_invade = np.max(score_invade_list, axis=0)
                score_surgery = np.max(score_surgery_list, axis=0)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        elif vote_mode.lower() in ['pred','pred_essential']:
            if mode.lower() in ['avg']:
                pred_invade = np.argmax(np.sum(pred_invade_list*pred_essential_list, axis=0))
                pred_surgery = np.argmax(np.sum(pred_surgery_list*pred_essential_list, axis=0))
                score_invade = np.sum(score_invade_list*pred_essential_list, axis=0)/(np.sum(pred_essential_list)+1e-5)
                score_surgery = np.sum(score_surgery_list*pred_essential_list, axis=0)/(np.sum(pred_essential_list)+1e-5)
            elif mode.lower() in ['max']:
                pred_invade = np.argmax(np.max(pred_invade_list*pred_essential_list, axis=0))
                pred_surgery = np.argmax(np.max(pred_surgery_list*pred_essential_list, axis=0))
                score_invade = np.max(score_invade_list*pred_essential_list, axis=0)/(np.max(pred_essential_list)+1e-5)
                score_surgery = np.max(score_surgery_list*pred_essential_list, axis=0)/(np.max(pred_essential_list)+1e-5)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        elif vote_mode.lower() in ['label', 'label_essential']:
            if mode.lower() in ['avg']:
                pred_invade = np.argmax(np.sum(pred_invade_list*label_essential_list, axis=0))
                pred_surgery = np.argmax(np.sum(pred_surgery_list*label_essential_list, axis=0))
                score_invade = np.sum(score_invade_list*label_essential_list, axis=0)/(np.sum(label_essential_list)+1e-5)
                score_surgery = np.sum(score_surgery_list*label_essential_list, axis=0)/(np.sum(label_essential_list)+1e-5)
            elif mode.lower() in ['max']:
                pred_invade = np.argmax(np.max(pred_invade_list*label_essential_list, axis=0))
                pred_surgery = np.argmax(np.max(pred_surgery_list*label_essential_list, axis=0))
                score_invade = np.max(score_invade_list*label_essential_list, axis=0)/(np.max(label_essential_list)+1e-5)
                score_surgery = np.max(score_surgery_list*label_essential_list, axis=0)/(np.max(label_essential_list)+1e-5)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        elif vote_mode.lower() in ['score','score_essential']:
            if mode.lower() in ['avg']:
                pred_invade = np.argmax(np.sum(pred_invade_list*score_essential_list, axis=0))
                pred_surgery = np.argmax(np.sum(pred_surgery_list*score_essential_list, axis=0))
                score_invade = np.sum(score_invade_list*score_essential_list, axis=0)/(np.sum(score_essential_list)+1e-5)
                score_surgery = np.sum(score_surgery_list*score_essential_list, axis=0)/(np.sum(score_essential_list)+1e-5)
            elif mode.lower() in ['max']:
                pred_invade = np.argmax(np.max(pred_invade_list*score_essential_list, axis=0))
                pred_surgery = np.argmax(np.max(pred_surgery_list*score_essential_list, axis=0))
                score_invade = np.max(score_invade_list*score_essential_list, axis=0)/(np.max(score_essential_list)+1e-5)
                score_surgery = np.max(score_surgery_list*score_essential_list, axis=0)/(np.max(score_essential_list)+1e-5)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        else:
            raise NotImplementedError("Voting mode: {} is not implemented!".format(vote_mode))
        label_invade = label_invade_list[0]
        label_surgery = label_surgery_list[0]
        
        preds_invade.append(pred_invade)
        preds_surgery.append(pred_surgery)
        preds_essential += pred_essential_list[:,0].tolist()
        scores_invade.append(score_invade)
        scores_surgery.append(score_surgery)
        scores_essential += score_essential_list.tolist()
        labels_invade.append(label_invade)
        labels_surgery.append(label_surgery)
        labels_essential += label_essential_list[:,0].tolist()
        
    return [preds_invade, preds_surgery, preds_essential],\
            [np.array(scores_invade), np.array(scores_surgery), scores_essential],\
            [labels_invade, labels_surgery, labels_essential]

def save_results(args, save_dict, output_path):
    results_dict = {}
    feature_dict = {}
    pred_invade_dict = save_dict['pred_invade_dict']
    pred_surgery_dict = save_dict['pred_surgery_dict']
    pred_essential_dict =save_dict['pred_essential_dict']
    score_invade_dict =save_dict['score_invade_dict']
    score_surgery_dict =save_dict['score_surgery_dict']
    score_essential_dict =save_dict['score_essential_dict']
    label_invade_dict =save_dict['label_invade_dict']
    label_surgery_dict =save_dict['label_surgery_dict']
    label_essential_dict =save_dict['label_essential_dict']
    if 'feat_dict' in list(save_dict.keys()):
        feat_dict = save_dict['feat_dict']
    for anno_item, item_dict in pred_invade_dict.items():
        results_dict[anno_item] = {}
        feature_dict[anno_item] = {}
        for img_name, v in item_dict.items():
            results_dict[anno_item][img_name]={
                'pred_invade': int(pred_invade_dict[anno_item][img_name]),
                'pred_surgery': int(pred_surgery_dict[anno_item][img_name]),
                'pred_essential': int(pred_essential_dict[anno_item][img_name]),
                'score_invade': score_invade_dict[anno_item][img_name],
                'score_surgery': score_surgery_dict[anno_item][img_name],
                'score_essential': score_essential_dict[anno_item][img_name],
                'label_invade': label_invade_dict[anno_item][img_name],
                'label_surgery': label_surgery_dict[anno_item][img_name],
                'label_essential': label_essential_dict[anno_item][img_name],
            }
            if 'feat_dict' in list(save_dict.keys()):
                feature_dict[anno_item][img_name] = feat_dict[anno_item][img_name]
            
    json.dump(results_dict, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    if 'feat_dict' in list(save_dict.keys()):
        pickle.dump(feature_dict, open(output_path.replace('.json','.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
        
def save_results_post(args, output_path, anno_item_list, preds, scores, labels):
    save_dict = {}
    for ii, anno_item in enumerate(anno_item_list):
        save_dict[anno_item] = {
            'pred_invade': int(preds[0][ii]),
            'pred_surgery': int(preds[1][ii]),
            'score_invade': scores[0][ii],
            'score_surgery': scores[1][ii],
            'label_invade': int(labels[0][ii]),
            'label_surgery': int(labels[1][ii])}
        
    json.dump(save_dict, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

def vote_w_esse_infer(args, pred_invade_dict, pred_surgery_dict, pred_essential_dict, 
                score_invade_dict, score_surgery_dict, score_essential_dict,
                mode = 'max'):
    vote_mode = args.vote_mode
    preds_invade, preds_surgery, preds_essential = [], [], []
    scores_invade, scores_surgery, scores_essential = [], [], []
    
    for anno_item in list(pred_invade_dict.keys()):
        pred_invade_list = np.array([v for k,v in pred_invade_dict[anno_item].items()])
        pred_surgery_list = np.array([v for k,v in pred_surgery_dict[anno_item].items()])
        pred_essential_list = np.array([v for k,v in pred_essential_dict[anno_item].items()])
        score_invade_list = np.array([v for k,v in score_invade_dict[anno_item].items()])
        score_surgery_list = np.array([v for k,v in score_surgery_dict[anno_item].items()])
        score_essential_list = np.array([v for k,v in score_essential_dict[anno_item].items()])

        if vote_mode.lower() in ['none']:
            if mode.lower() in ['avg']:
                pred_invade = 1 if np.mean(pred_invade_list)>=0.5 else 0
                pred_surgery = 1 if np.mean(pred_surgery_list)>=0.5 else 0
                score_invade = np.mean(pred_invade_list)
                score_surgery = np.mean(score_surgery_list)
            elif mode.lower() in ['max']:
                pred_invade = 1 if np.max(pred_invade_list)>=0.5 else 0
                pred_surgery = 1 if np.max(pred_surgery_list)>=0.5 else 0
                score_invade = np.max(pred_invade_list)
                score_surgery = np.max(score_surgery_list)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        elif vote_mode.lower() in ['pred','pred_essential']:
            if mode.lower() in ['avg']:
                pred_invade = 1 if np.sum(pred_invade_list*pred_essential_list)/(np.sum(pred_essential_list)+1e-5)>=0.5 else 0
                pred_surgery = 1 if np.sum(pred_surgery_list*pred_essential_list)/(np.sum(pred_essential_list)+1e-5)>=0.5 else 0
                score_invade = np.sum(score_invade_list*pred_essential_list)/(np.sum(pred_essential_list)+1e-5)
                score_surgery = np.sum(score_surgery_list*pred_essential_list)/(np.sum(pred_essential_list)+1e-5)
            elif mode.lower() in ['max']:
                pred_invade = 1 if np.max((np.array(pred_invade_list)>=0.5)*(np.array(pred_essential_list)>=0.5)*1) else 0
                pred_surgery = 1 if np.max((np.array(pred_surgery_list)>=0.5)*(np.array(pred_essential_list)>=0.5)*1) else 0
                score_invade = np.max(score_invade_list*pred_essential_list)/(np.max(pred_essential_list)+1e-5)
                score_surgery = np.max(score_surgery_list*pred_essential_list)/(np.max(pred_essential_list)+1e-5)
            else:
                raise NotImplementedError('Mode: {} is not Implemented!'.format(mode))
        else:
            raise NotImplementedError("Voting mode: {} is not implemented!".format(vote_mode))
        
        preds_invade.append(pred_invade) #
        preds_surgery.append(pred_surgery) #
        preds_essential += pred_essential_list.tolist() # 
        scores_invade.append(score_invade)  # out
        scores_surgery.append(score_surgery) # out
        scores_essential += score_essential_list.tolist() # choose top3

    return [preds_invade, preds_surgery, preds_essential],\
            [scores_invade, scores_surgery, scores_essential]
            
def save_results_infer(args, save_dict, output_path):
    results_dict = {}
    feature_dict = {}
    pred_invade_dict = save_dict['pred_invade_dict']
    pred_surgery_dict = save_dict['pred_surgery_dict']
    pred_essential_dict =save_dict['pred_essential_dict']
    score_invade_dict =save_dict['score_invade_dict']
    score_surgery_dict =save_dict['score_surgery_dict']
    score_essential_dict =save_dict['score_essential_dict']
    if 'feat_dict' in list(save_dict.keys()):
        feat_dict = save_dict['feat_dict']
    for anno_item, item_dict in pred_invade_dict.items():
        results_dict[anno_item] = {}
        feature_dict[anno_item] = {}
        for img_name, v in item_dict.items():
            results_dict[anno_item][img_name]={
                'pred_invade': int(pred_invade_dict[anno_item][img_name]),
                'pred_surgery': int(pred_surgery_dict[anno_item][img_name]),
                'pred_essential': int(pred_essential_dict[anno_item][img_name]),
                'score_invade': score_invade_dict[anno_item][img_name],
                'score_surgery': score_surgery_dict[anno_item][img_name],
                'score_essential': score_essential_dict[anno_item][img_name],
            }
            if 'feat_dict' in list(save_dict.keys()):
                feature_dict[anno_item][img_name] = feat_dict[anno_item][img_name]
            
    json.dump(results_dict, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    if 'feat_dict' in list(save_dict.keys()):
        pickle.dump(feature_dict, open(output_path.replace('.json','.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
        
        

def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label

def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics

def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics

def get_mIoU_metrics(args):
    results_dir = os.path.join(args.workspace, args.results_val, 'pred')
    semantic_dir = args.semantic_dir
    num_classes = args.net_num_classes
    aAcc = []
    IoU = []
    Acc = []
    for ii, result_name in enumerate(os.listdir(results_dir)):
        print("Calculating the {}/{} results....".format(ii,len(os.listdir(results_dir))), end='\r')
        result_path = os.path.join(results_dir, result_name)
        semantic_path = os.path.join(semantic_dir, result_name)
        result = cv2.imread(result_path, flags=0)
        result[result==128] = 1
        semantic = cv2.imread(semantic_path, flags=0)
        semantic[semantic==128] = 1
        ret_metrics = eval_metrics(result,
                    semantic,
                    num_classes,
                    ignore_index=255,
                    metrics=['mIoU'],
                    nan_to_num=None,
                    label_map=dict(),
                    reduce_zero_label=False,
                    beta=1)
        aAcc.append(ret_metrics['aAcc'])
        IoU.append(ret_metrics['IoU'])
        Acc.append(ret_metrics['Acc'])
        # print(ret_metrics)
    aAcc = np.mean(aAcc)
    IoU = np.mean(np.stack(IoU), axis=0)
    Acc = np.mean(np.stack(Acc), axis=0)

    return aAcc, IoU, Acc
