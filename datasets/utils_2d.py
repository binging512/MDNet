import cv2
import os
import numpy as np
import pydicom
import random
import torch

def read_img_n_label_files(args, item_dict, mode):
    img_path = item_dict['image_path']
    label_path = item_dict['seg_label_path']
    img = cv2.imread(img_path, flags=0)
    label = cv2.imread(label_path, flags=0)
    if args.neighbour_slice:
        prev_img = cv2.imread(item_dict['prev_image_path'], flags=0)
        next_img = cv2.imread(item_dict['next_image_path'], flags=0)
        img = np.stack((prev_img, img, next_img),axis=-1)
    else:
        img = np.stack((img,img,img), axis=-1)
    return img, label

def read_dcm_n_label_files(args, item_dict, mode):
    dcm_path = item_dict['dcm_path']
    label_path = item_dict['seg_label_path']
    # Read img and label files
    dcm_file = pydicom.dcmread(dcm_path)
    dcm = dcm_file.pixel_array
    label = cv2.imread(label_path, flags=0)
    if args.neighbour_slice:
        prev_dcm = pydicom.dcmread(item_dict['prev_dcm_path']).pixel_array
        next_dcm = pydicom.dcmread(item_dict['next_dcm_path']).pixel_array
        dcm = np.stack((prev_dcm, dcm, next_dcm),axis=-1)
    else:
        dcm = np.stack((dcm, dcm, dcm), axis=-1)
    
    # Preprocessing the dicom pixel array
    rescale_intercept = float(dcm_file.RescaleIntercept)
    rescale_slope = float(dcm_file.RescaleSlope)
    dcm = dcm*rescale_slope+rescale_intercept
    Win_center, Win_width = args.dcm_win_center, args.dcm_win_width
    low_window = Win_center - (Win_width/2)
    high_window = Win_center + (Win_width/2)
    dcm = np.clip(dcm, low_window, high_window).astype(np.float32)
    dcm = (dcm-low_window)/(high_window-low_window)*255
    dcm=dcm.astype(np.uint8)
        
    return dcm, label

def get_essential_label(args, item_dict, anno):
    if args.net_essential_object.lower() in ['esse', 'essential']:
        essential_start = anno['essential_layers']['start']
        essential_stop = anno['essential_layers']['stop']
    elif args.net_essential_object.lower() in ['tumor']:
        essential_start = anno['tumor_layers']['start']
        essential_stop = anno['tumor_layers']['stop']
    elif args.net_essential_object.lower() in ['vein']:
        essential_start = anno['vein_layers']['start']
        essential_stop = anno['vein_layers']['stop']
    elif args.net_essential_object.lower() in ['tumor_vein']:
        essential_start = min([anno['tumor_layers']['start'], anno['vein_layers']['start']])
        essential_stop = min([anno['tumor_layers']['stop'], anno['vein_layers']['stop']])
    else:
        raise NotImplementedError("net_essential_object {} is not implemented!".format(args.net_essential_object))
    item_num = int(os.path.basename(item_dict['image_path']).split('img')[-1].split('.')[0])
    if item_num in range(essential_start, essential_stop):
        essential_label = 1
    else:
        essential_label = 0
    return essential_label

def resize(args, img, label):
    img = cv2.resize(img, args.rescale, interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, args.rescale, interpolation=cv2.INTER_NEAREST)
    return img, label

def random_crop(args, img, label):
    H,W,C = img.shape
    # Random crop
    h_start = random.randrange(0, H-args.crop_size[1])
    w_start = random.randrange(0, W-args.crop_size[0])
    h_end = h_start + args.crop_size[1]
    w_end = w_start + args.crop_size[0]
    img = img[h_start:h_end, w_start:w_end, :]
    label = label[h_start:h_end, w_start:w_end]
    
    return img, label

def center_crop(args, img, label):
    H,W,C = img.shape
    # center crop
    h_start = round((H-args.crop_size[1])/2)
    w_start = round((W-args.crop_size[0])/2)
    h_end = h_start+args.crop_size[1]
    w_end = w_start+args.crop_size[0]
    img = img[h_start:h_end, w_start:w_end, :]
    label = label[h_start:h_end, w_start:w_end]
    
    return img, label

def random_crop_with_window(args, img, label):
    H,W,C = img.shape
    # Center crop the random window
    win_h_start = round((H-args.crop_window[1])/2)
    win_w_start = round((W-args.crop_window[0])/2)
    # Random Crop
    relative_h_start = random.randrange(0, args.crop_window[1]-args.crop_size[1])
    relative_w_start = random.randrange(0, args.crop_window[0]-args.crop_size[0])
    h_start = win_h_start + relative_h_start
    w_start = win_w_start + relative_w_start
    h_end = h_start + args.crop_size[1]
    w_end = w_start + args.crop_size[0]
    # Crop the image and label
    img = img[h_start:h_end, w_start:w_end, :]
    label = label[h_start:h_end, w_start:w_end]
    return img, label

def read_img_meta(args, anno, mode):
    patient_id = anno['patient_id']
    # Blood Measurement
    # Hb = anno['Hb']
    # WBC = anno['WBC']
    # NEUT = anno['NEUT']
    # PLT = anno['PLT']
    # ALB = anno["ALB"]
    # ALT = anno["ALT"]
    # AST = anno["AST"]
    # TB = anno["TB"]
    # DB = anno["DB"]
    # GGT = anno["GGT"]
    # CA199 = anno["CA199"]
    # CEA = anno["CEA"]
    # Normalized
    Hb = (anno['Hb']-100)/10
    WBC = (anno['WBC']-5)
    NEUT = (anno['NEUT']-50)/10
    PLT = (anno['PLT']-200)/100
    ALB = (anno["ALB"]-30)/10
    ALT = (anno["ALT"]-80)/100
    AST = (anno["AST"]-50)/100
    TB = (anno["TB"]-90)/100
    DB = (anno["DB"]-70)/100
    GGT = (anno["GGT"]-350)/100
    CA199 = (anno["CA199"]-1000)/1000
    CEA = (anno["CEA"]-10)/10
    blood_input = torch.tensor([Hb, WBC, NEUT, PLT, ALB, ALT, AST, TB, DB, GGT, CEA, CA199], dtype=torch.float32)
    blood_des = anno['blood_des']
    
    # other information
    if anno['sex']=='female':
        sex, sex_prompt = 0, ['She', 'Her']
    elif anno['sex']=='male':
        sex, sex_prompt = 1, ['He', 'His']
    symptom = anno["symptom"]
    age = anno["age"]/100
    smoke = anno['smoke']
    diabetes = anno["diabetes"]
    heart_brain_blood_vessel = anno['heart_brain_blood_vessel']
    family = anno['family']
    others_input = torch.tensor([sex, symptom, age, smoke, diabetes, heart_brain_blood_vessel, family], dtype=torch.float32)
    others_des = anno['others_des']
    
    img_meta = {
        'patient_id': patient_id,
        "blood": blood_input, 
        "others": others_input,
        "blood_des": blood_des,
        "others_des": others_des,
        }
    return img_meta

def read_img_meta_mask(args, anno, mode):
    patient_id = anno['patient_id']
    # Blood Measurement
    # Hb = anno['Hb']
    # WBC = anno['WBC']
    # NEUT = anno['NEUT']
    # PLT = anno['PLT']
    # ALB = anno["ALB"]
    # ALT = anno["ALT"]
    # AST = anno["AST"]
    # TB = anno["TB"]
    # DB = anno["DB"]
    # GGT = anno["GGT"]
    # CA199 = anno["CA199"]
    # CEA = anno["CEA"]
    # Normalized
    Hb = (anno['Hb']-100)/10
    WBC = (anno['WBC']-5)
    NEUT = (anno['NEUT']-50)/10
    PLT = (anno['PLT']-200)/100
    ALB = (anno["ALB"]-30)/10
    ALT = (anno["ALT"]-80)/100
    AST = (anno["AST"]-50)/100
    TB = (anno["TB"]-90)/100
    DB = (anno["DB"]-70)/100
    GGT = (anno["GGT"]-350)/100
    CA199 = (anno["CA199"]-1000)/1000
    CEA = (anno["CEA"]-10)/10
    blood_input = [Hb, WBC, NEUT, PLT, ALB, ALT, AST, TB, DB, GGT, CEA, CA199]
    
    # other information
    if anno['sex']=='female':
        sex, sex_prompt = 0, ['She', 'Her']
    elif anno['sex']=='male':
        sex, sex_prompt = 1, ['He', 'His']
    symptom = anno["symptom"]
    age = anno["age"]/100
    smoke = anno['smoke']
    diabetes = anno["diabetes"]
    heart_brain_blood_vessel = anno['heart_brain_blood_vessel']
    family = anno['family']
    others_input = [sex, symptom, age, smoke, diabetes, heart_brain_blood_vessel, family]
    
    
    if mode in ['train', 'all']:
        mask_rate = args.train_mask_rate
        blood_mask_indice = random.sample(range(len(blood_input)),round(len(blood_input)*mask_rate))
        for idx in blood_mask_indice:
            blood_input[idx] = 0
        blood_input = torch.tensor(blood_input, dtype=torch.float32)
        blood_des = "This is a CT image of a patient. {} blood test report is as follows: \
                    {} hemoglobin level is {}; \
                    {} white cell level is {}; \
                    {} neutrophil percentage is {}%; \
                    {} platelets level is {}; \
                    {} albumin level is {}; \
                    {} alanine aminotransferase level is {}; \
                    {} aspartate aminotransferase level is {}; \
                    {} total bilirubin level is {}; \
                    {} direct bilirubin level is {}; \
                    {} gamma-glutamyl transpeptidase level is {}; \
                    {} carcinoembryonic antigen level is {}; \
                    {} CA199 level is {}; \
                        ".format(
                            sex_prompt[1],
                            sex_prompt[1], str(Hb),
                            sex_prompt[1], str(WBC),
                            sex_prompt[1], str(NEUT),
                            sex_prompt[1], str(PLT),
                            sex_prompt[1], ALB,
                            sex_prompt[1], ALT,
                            sex_prompt[1], AST,
                            sex_prompt[1], TB,
                            sex_prompt[1], DB,
                            sex_prompt[1], GGT,
                            sex_prompt[1], CEA,
                            sex_prompt[1], CA199,
                        )
        blood_des = ' '.join(blood_des.split())
        
    elif mode in ['val','test']:
        pass
    
    others_input = torch.tensor([sex, symptom, age, smoke, diabetes, heart_brain_blood_vessel, family], dtype=torch.float32)
    
    blood_des = anno['blood_des']
    others_des = anno['others_des']
    
    img_meta = {
        'patient_id': patient_id,
        "blood": blood_input, 
        "others": others_input,
        "blood_des": blood_des,
        "others_des": others_des,
        }
    return img_meta

def detection_crop(args, img, label, anno):
    x1 = anno['det_bbox']['x_min']
    y1 = anno['det_bbox']['y_min']
    x2 = anno['det_bbox']['x_max']
    y2 = anno['det_bbox']['y_max']
    box_H = y2-y1
    box_W = x2-x1
    if box_H>box_W:
        diff = box_H-box_W
        crp_x1 = max(0, x1-int(diff/2))
        crp_x2 = crp_x1 + box_H
        crp_y1, crp_y2 = y1, y2
    else:
        diff = box_W-box_H
        crp_y1 = max(0, y1-int(diff/2))
        crp_y2 = crp_y1 + box_W
        crp_x1, crp_x2 = x1, x2
    img = img[crp_y1:crp_y2, crp_x1:crp_x2]
    label = label[crp_y1:crp_y2, crp_x1:crp_x2]
    return img, label

def get_esse_list(args, item_list, anno_dict):
    esse_list = []
    for item_dict in item_list:
        anno = anno_dict[item_dict['anno_item']]
        if args.net_essential_object.lower() in ['esse', 'essential']:
            essential_start = anno['essential_layers']['start']
            essential_stop = anno['essential_layers']['stop']
        elif args.net_essential_object.lower() in ['tumor']:
            essential_start = anno['tumor_layers']['start']
            essential_stop = anno['tumor_layers']['stop']
        elif args.net_essential_object.lower() in ['vein']:
            essential_start = anno['vein_layers']['start']
            essential_stop = anno['vein_layers']['stop']
        elif args.net_essential_object.lower() in ['tumor_vein']:
            essential_start = min([anno['tumor_layers']['start'], anno['vein_layers']['start']])
            essential_stop = min([anno['tumor_layers']['stop'], anno['vein_layers']['stop']])
        else:
            raise NotImplementedError("net_essential_object {} is not implemented!".format(args.net_essential_object))
        item_num = int(os.path.basename(item_dict['image_path']).split('img')[-1].split('.')[0])
        if item_num in range(essential_start, essential_stop):
            essential_label = 1
        else:
            essential_label = 0
        esse_list.append(essential_label)
    return esse_list

def recollect_item(mode, item_list, anno_dict):
    new_item_list = []
    for item in item_list:
        anno_item = item["anno_item"]
        item_info = anno_dict[anno_item]
        img_idx = int(os.path.basename(item["image_path"]).split('img')[-1].split('.jpg')[0])
        if mode in ['train']:
            slice_start = min(item_info["tumor_layers"]["start"], item_info["det_bbox"]["z_min"])
            slice_stop = max(item_info["tumor_layers"]["stop"], item_info["det_bbox"]["z_max"])
        else:
            slice_margin = int(0.05*(item_info["det_bbox"]["z_max"]-item_info["det_bbox"]["z_min"]))
            slice_start = max(0, item_info["det_bbox"]["z_min"]-slice_margin)
            slice_stop = item_info["det_bbox"]["z_max"]+slice_margin
        if img_idx >= slice_start and img_idx<= slice_stop:
            new_item_list.append(item)
    return new_item_list