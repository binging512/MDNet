import json
import pickle

def save_results(args, label_invade_list, label_surgery_list,
                 anno_item_list, score_invade_list, score_surgery_list, pred_invade_list, pred_surgery_list,
                 ct_score_invade_list, ct_score_surgery_list, ct_pred_invade_list, ct_pred_surgery_list, 
                 output_path):
    results_dict = {}
    for ii, anno_item in enumerate(anno_item_list):
        results_dict[anno_item] = {
            'label_invade': label_invade_list[ii],
            'label_surgery': label_surgery_list[ii],
            'score_invade': score_invade_list[ii],
            'score_surgery': score_surgery_list[ii],
            'pred_invade': int(pred_invade_list[ii]),
            'pred_surgery': int(pred_surgery_list[ii]),
            'ct_score_invade': ct_score_invade_list[ii],
            'ct_score_surgery': ct_score_surgery_list[ii],
            'ct_pred_invade': int(ct_pred_invade_list[ii]),
            'ct_pred_surgery': int(ct_pred_surgery_list[ii]),
        }
    json.dump(results_dict, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    


def save_results_test(args, label_invade_list, label_surgery_list,
                 anno_item_list, score_invade_list, score_surgery_list, pred_invade_list, pred_surgery_list,
                 ct_score_invade_list, ct_score_surgery_list, ct_pred_invade_list, ct_pred_surgery_list, 
                 img_feat_list, output_path):
    results_dict = {}
    feature_dict = {}
    for ii, anno_item in enumerate(anno_item_list):
        results_dict[anno_item] = {
            'label_invade': label_invade_list[ii],
            'label_surgery': label_surgery_list[ii],
            'score_invade': score_invade_list[ii],
            'score_surgery': score_surgery_list[ii],
            'pred_invade': int(pred_invade_list[ii]),
            'pred_surgery': int(pred_surgery_list[ii]),
            'ct_score_invade': ct_score_invade_list[ii],
            'ct_score_surgery': ct_score_surgery_list[ii],
            'ct_pred_invade': int(ct_pred_invade_list[ii]),
            'ct_pred_surgery': int(ct_pred_surgery_list[ii]),
        }
        feature_dict[anno_item] = {
            'label_invade': label_invade_list[ii],
            'label_surgery': label_surgery_list[ii],
            'score_invade': score_invade_list[ii],
            'score_surgery': score_surgery_list[ii],
            'pred_invade': int(pred_invade_list[ii]),
            'pred_surgery': int(pred_surgery_list[ii]),
            'ct_score_invade': ct_score_invade_list[ii],
            'ct_score_surgery': ct_score_surgery_list[ii],
            'ct_pred_invade': int(ct_pred_invade_list[ii]),
            'ct_pred_surgery': int(ct_pred_surgery_list[ii]),
            'img_feature': img_feat_list[ii],
        }
    json.dump(results_dict, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    pickle.dump(feature_dict, open(output_path.replace('.json','.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    