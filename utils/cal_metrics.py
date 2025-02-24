import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize


def cal_class_metrics(args, logger, preds, scores, labels, val_thres_inv=0.5, val_thres_sur=0.5):
    # AUC
    auc_invade = roc_auc_score(labels[0], scores[0])
    invade_aucs = '-'
    auc_surgery = roc_auc_score(labels[1], scores[1])
    surgery_aucs = '-'
    auc_essential = roc_auc_score(labels[2], scores[2])
    
    # FPR, TPR
    fpr_invade, tpr_invade, thres_invade = roc_curve(labels[0], scores[0])
    fpr_surgery, tpr_surgery, thres_surgery = roc_curve(labels[1], scores[1])
    fpr_essential, tpr_essential, thres_essential = roc_curve(labels[2], scores[2])
    
    sen_invade, spe_invade = tpr_invade, 1-fpr_invade
    sen_surgery, spe_surgery = tpr_surgery, 1-fpr_surgery
    sen_essential, spe_essential = tpr_essential, 1-fpr_essential
    best_idx_invade = np.argmax(sen_invade*spe_invade)
    best_idx_surgery = np.argmax(sen_surgery*spe_surgery)
    best_idx_essential = np.argmax(sen_essential*spe_essential)
    best_sen_invade, best_spe_invade, best_thres_invade = sen_invade[best_idx_invade], spe_invade[best_idx_invade], thres_invade[best_idx_invade]
    best_sen_surgery, best_spe_surgery, best_thres_surgery = sen_surgery[best_idx_surgery], spe_surgery[best_idx_surgery], thres_surgery[best_idx_surgery]
    best_sen_essential, best_spe_essential, best_thres_essential = sen_essential[best_idx_essential], spe_essential[best_idx_essential], thres_essential[best_idx_essential]
    
    # best f1
    best_pred_invade = [1 if s>=best_thres_invade else 0 for s in scores[0]]
    best_pred_surgery = [1 if s>=best_thres_surgery else 0 for s in scores[1]]
    best_f1_invade = f1_score(labels[0], best_pred_invade)
    best_f1_surgery = f1_score(labels[1], best_pred_surgery)
    
    # Best precision 
    best_prec_invade = precision_score(labels[0], best_pred_invade)
    best_prec_surgery = precision_score(labels[1], best_pred_surgery)
    best_recall_invade = recall_score(labels[0], best_pred_invade)
    best_recall_surgery = recall_score(labels[1], best_pred_surgery)
    best_acc_invade = accuracy_score(labels[0], best_pred_invade)
    best_acc_surgery = accuracy_score(labels[1], best_pred_surgery)
    
    # F1
    f1_invade = f1_score(labels[0], preds[0], average='macro')
    f1_surgery = f1_score(labels[1], preds[1], average='macro')
    f1_essential = f1_score(labels[2], preds[2])
    # Precision
    prec_invade = precision_score(labels[0], preds[0])
    prec_surgery = precision_score(labels[1], preds[1])
    prec_essential = precision_score(labels[2], preds[2])
    # Recall
    recall_invade = recall_score(labels[0], preds[0])
    recall_surgery = recall_score(labels[1], preds[1])
    recall_essential = recall_score(labels[2], preds[2])
    # Acc
    acc_invade = accuracy_score(labels[0], preds[0])
    acc_surgery = accuracy_score(labels[1], preds[1])
    acc_essential = accuracy_score(labels[2], preds[2])
    # Confusion Matrix
    conf_invade = confusion_matrix(labels[0], preds[0])
    conf_surgery = confusion_matrix(labels[1], preds[1])
    conf_essential = confusion_matrix(labels[2], preds[2])
    
    # Sensitivity & Specificity
    TP_invade = np.sum(np.array(labels[0]) * np.array(preds[0]))
    TN_invade = np.sum((1-np.array(labels[0])) * (1-np.array(preds[0])))
    FP_invade = np.sum(np.array(labels[0]) * (1-np.array(preds[0])))
    FN_invade = np.sum((1-np.array(labels[0])) * np.array(preds[0]))
    sen_invade_pro = TP_invade/(TP_invade+FN_invade)
    spe_invade_pro = TN_invade/(TN_invade+FP_invade)
    TP_surgery = np.sum(np.array(labels[1]) * np.array(preds[1]))
    TN_surgery = np.sum((1-np.array(labels[1])) * (1-np.array(preds[1])))
    FP_surgery = np.sum(np.array(labels[1]) * (1-np.array(preds[1])))
    FN_surgery = np.sum((1-np.array(labels[1])) * np.array(preds[1]))
    sen_surgery_pro = TP_surgery/(TP_surgery+FN_surgery)
    spe_surgery_pro = TN_surgery/(TN_surgery+FP_surgery)
    
    # val Sensitivity & Specificity & f1-score
    val_pred_invade = [1 if s>=val_thres_inv else 0 for s in scores[0]]
    val_pred_surgery = [1 if s>=val_thres_sur else 0 for s in scores[1]]
    
    val_f1_invade = f1_score(labels[0], val_pred_invade)
    val_f1_surgery = f1_score(labels[1], val_pred_surgery)
    val_TP_invade = np.sum(np.array(labels[0]) * np.array(val_pred_invade))
    val_TN_invade = np.sum((1-np.array(labels[0])) * (1-np.array(val_pred_invade)))
    val_FP_invade = np.sum(np.array(labels[0]) * (1-np.array(val_pred_invade)))
    val_FN_invade = np.sum((1-np.array(labels[0])) * np.array(val_pred_invade))
    val_sen_invade = val_TP_invade/(val_TP_invade+val_FN_invade)
    val_spe_invade = val_TN_invade/(val_TN_invade+val_FP_invade)
    val_TP_surgery = np.sum(np.array(labels[1]) * np.array(val_pred_surgery))
    val_TN_surgery = np.sum((1-np.array(labels[1])) * (1-np.array(val_pred_surgery)))
    val_FP_surgery = np.sum(np.array(labels[1]) * (1-np.array(val_pred_surgery)))
    val_FN_surgery = np.sum((1-np.array(labels[1])) * np.array(val_pred_surgery))
    val_sen_surgery = val_TP_surgery/(val_TP_surgery+val_FN_surgery)
    val_spe_surgery = val_TN_surgery/(val_TN_surgery+val_FP_surgery)
    
    
    logger.info("Classification Results:")
    logger.info("AUC Invade: {:.5f}. Details: {}".format(auc_invade, invade_aucs))
    logger.info("AUC Surgery: {:.5f}. Details: {}".format(auc_surgery, surgery_aucs))
    logger.info("AUC Essential: {:.5f}".format(auc_essential))
    logger.info("Sen Invade: {:.5f}, Spe Invade: {:.5f}, Thres Invade: {:.5f}".format(best_sen_invade, best_spe_invade, best_thres_invade))
    logger.info("Sen Surgery: {:.5f}, Spe Surgery: {:.5f}, Thres Surgery: {:.5f}".format(best_sen_surgery, best_spe_surgery, best_thres_surgery))
    logger.info("Sen Essential: {:.5f}, Spe Essential: {:.5f}, Thres Essential: {:.5f}".format(best_sen_essential, best_spe_essential, best_thres_essential))
    logger.info("Sen Invade: {:.5f}, Spe Invade: {:.5f}, Thres Invade: {}".format(sen_invade_pro, spe_invade_pro, 0.5))
    logger.info("Sen Surgery: {:.5f}, Spe Surgery: {:.5f}, Thres Surgery: {}".format(sen_surgery_pro, spe_surgery_pro, 0.5))
    
    logger.info("F1_score Invade: {:.5f}, Best F1_score Invade: {:.5f}".format(f1_invade, best_f1_invade))
    logger.info("F1_score Surgery: {:.5f}, Best F1_score Surgery: {:.5f}".format(f1_surgery, best_f1_surgery))
    logger.info("F1_score Essential: {:.5f}".format(f1_essential))
    logger.info("Prec Invade: {:.5f}, Rec Invade: {:.5f}, Acc Invade: {:.5f}".format(prec_invade, recall_invade, acc_invade))
    logger.info("Prec Surgery: {:.5f}, Rec Surgery: {:.5f}, Acc Surgery: {:.5f}".format(prec_surgery, recall_surgery, acc_surgery))
    logger.info("Prec Essential: {:.5f}, Rec Essential: {:.5f}, Acc Essential: {:.5f}".format(prec_essential, recall_essential, acc_essential))
    logger.info("Best Prec Invade: {:.5f}, Best Rec Invade: {:.5f}, Best Acc Invade: {:.5f}".format(best_prec_invade, best_recall_invade, best_acc_invade))
    logger.info("Best Prec Surgery: {:.5f}, Best Rec Surgery: {:.5f}, Best Acc Surgery: {:.5f}".format(best_prec_surgery, best_recall_surgery, best_acc_surgery))
    
    logger.info("Val Sen Invade: {:.5f}, Val Spe Invade: {:.5f}, Thres Invade: {}".format(val_sen_invade, val_spe_invade, val_thres_inv))
    logger.info("Val Sen Surgery: {:.5f}, Val Spe Surgery: {:.5f}, Thres Surgery: {}".format(val_sen_surgery, val_spe_surgery, val_thres_sur))
    logger.info("Val F1_score Invade: {:.5f}, Val F1_score Surgery: {:.5f}".format(val_f1_invade, val_f1_surgery))
    
    logger.info("Invade Confusion Matrix: {}".format(conf_invade))
    logger.info("Surgery Confusion Matrix: {}".format(conf_surgery))
    logger.info("Essential Confusion Matrix: {}".format(conf_essential))
    
    results = {
        "auc_invade": auc_invade, "auc_surgery": auc_surgery, "auc_essential": auc_essential,
        "sen_invade": best_sen_invade, "spe_invade": best_spe_invade, "thres_invade": best_thres_invade,
        "sen_surgery": best_sen_surgery, "spe_surgery": best_spe_surgery, "thres_surgery": best_thres_surgery,
        "sen_essential": best_sen_essential, "spe_essential": best_spe_essential, "thres_essential": best_thres_essential,
        'f1_invade': f1_invade, 'f1_surgery': f1_surgery, 'f1_essential': f1_essential,
        "prec_invade": prec_invade, "recall_invade": recall_invade, "acc_invade": acc_invade,
        "prec_surgery": prec_surgery, "recall_surgery": recall_surgery, "acc_surgery": acc_surgery,
        "prec_essential": prec_essential, "recall_essential": recall_essential, "acc_essential": acc_essential,
        "val_sen_invade": val_sen_invade, "val_spe_invade": val_spe_invade, 'val_f1_invade': val_f1_invade,
        "val_sen_surgery": val_sen_surgery, "val_spe_surgery": val_spe_surgery, 'val_f1_surgery': val_f1_surgery,
        "conf_invade": conf_invade, "conf_surgery": conf_surgery, "conf_essential": conf_essential,
    }
    return results

def cal_class_metrics_3d(args, logger, preds, scores, labels):
    # AUC
    auc_invade = roc_auc_score(labels[0], scores[0])
    invade_aucs = '-'
    auc_surgery = roc_auc_score(labels[1], scores[1])
    surgery_aucs = '-'
    
    # FPR, TPR
    fpr_invade, tpr_invade, thres_invade = roc_curve(labels[0], scores[0])
    fpr_surgery, tpr_surgery, thres_surgery = roc_curve(labels[1], scores[1])
    
    sen_invade, spe_invade = tpr_invade, 1-fpr_invade
    sen_surgery, spe_surgery = tpr_surgery, 1-fpr_surgery
    best_idx_invade = np.argmax(sen_invade*spe_invade)
    best_idx_surgery = np.argmax(sen_surgery*spe_surgery)
    best_sen_invade, best_spe_invade, best_thres_invade = sen_invade[best_idx_invade], spe_invade[best_idx_invade], thres_invade[best_idx_invade]
    best_sen_surgery, best_spe_surgery, best_thres_surgery = sen_surgery[best_idx_surgery], spe_surgery[best_idx_surgery], thres_surgery[best_idx_surgery]
    
    # best f1
    best_pred_invade = [1 if s>=best_thres_invade else 0 for s in scores[0]]
    best_pred_surgery = [1 if s>=best_thres_surgery else 0 for s in scores[1]]
    best_f1_invade = f1_score(labels[0], best_pred_invade)
    best_f1_surgery = f1_score(labels[1], best_pred_surgery)
    
    # F1
    f1_invade = f1_score(labels[0], preds[0], average='macro')
    f1_surgery = f1_score(labels[1], preds[1], average='macro')
    # Precision
    prec_invade = precision_score(labels[0], preds[0])
    prec_surgery = precision_score(labels[1], preds[1])
    # Recall
    recall_invade = recall_score(labels[0], preds[0])
    recall_surgery = recall_score(labels[1], preds[1])
    # Acc
    acc_invade = accuracy_score(labels[0], preds[0])
    acc_surgery = accuracy_score(labels[1], preds[1])
    # Confusion Matrix
    conf_invade = confusion_matrix(labels[0], preds[0])
    conf_surgery = confusion_matrix(labels[1], preds[1])
    
    # Sensitivity & Specificity
    TP_invade = np.sum(np.array(labels[0]) * np.array(preds[0]))
    TN_invade = np.sum((1-np.array(labels[0])) * (1-np.array(preds[0])))
    FP_invade = np.sum(np.array(labels[0]) * (1-np.array(preds[0])))
    FN_invade = np.sum((1-np.array(labels[0])) * np.array(preds[0]))
    sen_invade_pro = TP_invade/(TP_invade+FN_invade)
    spe_invade_pro = TN_invade/(TN_invade+FP_invade)
    TP_surgery = np.sum(np.array(labels[1]) * np.array(preds[1]))
    TN_surgery = np.sum((1-np.array(labels[1])) * (1-np.array(preds[1])))
    FP_surgery = np.sum(np.array(labels[1]) * (1-np.array(preds[1])))
    FN_surgery = np.sum((1-np.array(labels[1])) * np.array(preds[1]))
    sen_surgery_pro = TP_surgery/(TP_surgery+FN_surgery)
    spe_surgery_pro = TN_surgery/(TN_surgery+FP_surgery)
    
    logger.info("Classification Results:")
    logger.info("AUC Invade: {:.5f}. Details: {}".format(auc_invade, invade_aucs))
    logger.info("AUC Surgery: {:.5f}. Details: {}".format(auc_surgery, surgery_aucs))
    logger.info("Sen Invade: {:.5f}, Spe Invade: {:.5f}, Thres Invade: {:.5f}".format(best_sen_invade, best_spe_invade, best_thres_invade))
    logger.info("Sen Surgery: {:.5f}, Spe Surgery: {:.5f}, Thres Surgery: {:.5f}".format(best_sen_surgery, best_spe_surgery, best_thres_surgery))
    logger.info("Sen Invade: {:.5f}, Spe Invade: {:.5f}, Thres Invade: {}".format(sen_invade_pro, spe_invade_pro, 0.5))
    logger.info("Sen Surgery: {:.5f}, Spe Surgery: {:.5f}, Thres Surgery: {}".format(sen_surgery_pro, spe_surgery_pro, 0.5))
    
    logger.info("F1_score Invade: {:.5f}, Best F1_score Invade: {:.5f}".format(f1_invade, best_f1_invade))
    logger.info("F1_score Surgery: {:.5f}, Best F1_score Surgery: {:.5f}".format(f1_surgery, best_f1_surgery))
    logger.info("Prec Invade: {:.5f}, Rec Invade: {:.5f}, Acc Invade: {:.5f}".format(prec_invade, recall_invade, acc_invade))
    logger.info("Prec Surgery: {:.5f}, Rec Surgery: {:.5f}, Acc Surgery: {:.5f}".format(prec_surgery, recall_surgery, acc_surgery))
    
    logger.info("Invade Confusion Matrix: {}".format(conf_invade))
    logger.info("Surgery Confusion Matrix: {}".format(conf_surgery))
    
    results = {
        "auc_invade": auc_invade, "auc_surgery": auc_surgery,
        "sen_invade": best_sen_invade, "spe_invade": best_spe_invade, "thres_invade": best_thres_invade,
        "sen_surgery": best_sen_surgery, "spe_surgery": best_spe_surgery, "thres_surgery": best_thres_surgery,
        'f1_invade': f1_invade, 'f1_surgery': f1_surgery,
        "prec_invade": prec_invade, "recall_invade": recall_invade, "acc_invade": acc_invade,
        "prec_surgery": prec_surgery, "recall_surgery": recall_surgery, "acc_surgery": acc_surgery,
        "conf_invade": conf_invade, "conf_surgery": conf_surgery,
    }
    return results

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("Pancls training argument parser.")
    parser.add_argument('--net_invade_classes', default=2, type=int)
    parser.add_argument('--net_surgery_classes', default=2, type=int)
    args = parser.parse_args()
    preds = np.array([[0,0,0,1,1,1,1,1,0,0,0], [0,0,0,1,1,1,1,1,0,0,0], [0,0,0,1,1,1,1,1,0,0,0]])
    scores = np.array([[[1,0],[1,0],[0.9,0.1],[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.5,0.5],[0.4,0.6],[0.1,0.9],[0,1],[0,1]],
                       [[1,0],[1,0],[0.9,0.1],[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.5,0.5],[0.4,0.6],[0.1,0.9],[0,1],[0,1]],
                       [[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1]]])
    labels = np.array([[0,0,0,0,0,1,1,1,1,1,1],[0,0,0,0,0,1,1,1,1,1,1], [0,0,0,0,0,1,1,1,1,1,1]])
    # scores[]
    cal_class_metrics(args, None, preds, scores, labels)