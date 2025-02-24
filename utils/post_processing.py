import json
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score, recall_score ,f1_score

def post_processing(logger, label_invade_list, label_surgery_list, score_invade_list, score_surgery_list):
    post_invade_list, post_surgery_list = [], []
    for ii, score_invade in enumerate(score_invade_list):
        score_invade = score_invade_list[ii]
        score_surgery = score_surgery_list[ii]
        temp_invade = np.tan(score_invade*np.pi/2)
        temp_surgery = np.tan(score_surgery*np.pi/2)
        post_invade = np.tanh(score_invade*temp_surgery)
        post_surgery = np.tanh(score_surgery*temp_invade)
        post_invade_list.append(post_invade)
        post_surgery_list.append(post_surgery)
        
    post_auc_invade = roc_auc_score(label_invade_list, post_invade_list)
    post_auc_surgery = roc_auc_score(label_surgery_list, post_surgery_list)
    logger.info("======= Post-processing ========")
    logger.info('Post Invade AUC: {:.5f}'.format(post_auc_invade))
    logger.info('Post Surgery AUC: {:.5f}'.format(post_auc_surgery))
    
def offline_post_processing():
    results_path = 'workspace/3D_final_select/r2plus1d18_ep100_img_b8_rz360_win256_crp224_z64i1_0.4/results_test/results_test.json'
    results_dict = json.load(open(results_path,'r', encoding='utf-8'))
    label_invade_list = []
    label_surgery_list = []
    score_invade_list = []
    score_surgery_list = []
    post_invade_list = []
    post_surgery_list = []
    for k,v in results_dict.items():
        label_invade_list.append(v['label_invade'])
        label_surgery_list.append(v['label_surgery'])
        score_invade_list.append(v['score_invade'])
        score_surgery_list.append(v['score_surgery'])
        score_invade = v['score_invade']
        temp_invade = np.tan(score_invade*np.pi/2)
        score_surgery = v['score_surgery']
        temp_surgery = np.tan(score_surgery*np.pi/2)
        post_invade = np.tanh(score_invade*temp_surgery)
        post_invade_list.append(post_invade)
        post_surgery = np.tanh(score_surgery*temp_invade)
        post_surgery_list.append(post_surgery)
        
    auc_invade = roc_auc_score(label_invade_list, score_invade_list)
    post_auc_invade = roc_auc_score(label_invade_list, post_invade_list)
    auc_surgery = roc_auc_score(label_surgery_list, score_surgery_list)
    post_auc_surgery = roc_auc_score(label_surgery_list, post_surgery_list)
    print(auc_invade)
    print(post_auc_invade)
    print(auc_surgery)
    print(post_auc_surgery)
    

if __name__=="__main__":
    offline_post_processing()