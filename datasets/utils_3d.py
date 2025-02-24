import cv2
import os
import pydicom
import numpy as np
import torch
import torch.nn.functional as F
import random
import pickle
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode

def read_img_n_label_files(args, item_dict, mode):
    img_dir = item_dict['image_dir']
    label_dir = item_dict['seg_label_dir']
    img_names = sorted(os.listdir(img_dir))
    img_list = []
    label_list = []
    
    # Read img and label files
    for img_name in img_names:
        # Image
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path,flags=0)
        # Label
        label_path = os.path.join(label_dir, img_name.replace('.jpg','.png'))
        label = cv2.imread(label_path, flags=0) if os.path.exists(label_path) else np.zeros_like(img)
        img_list.append(img)
        label_list.append(label)
        
    img = np.stack(img_list, axis=0).astype(np.float32)
    label = np.stack(label_list, axis=0).astype(np.float32)
    # Set the slice_thickness to 1
    slice_thickness = item_dict['slice_thickness']
    D, H, W = img.shape
    dest_D = round(D*slice_thickness/1)
    # keep ratio rescaling
    dest_H,dest_W = args.rescale
    dest_D = round((dest_H/H)*dest_D)
    
    # Rescale
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    label = torch.tensor(label).unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, (dest_D, dest_H, dest_W), mode='trilinear').squeeze(0)/255
    label = F.interpolate(label, (dest_D, dest_H, dest_W), mode='nearest').squeeze(0)
    return img, label

def read_dcm_n_label_files(args, item_dict, mode):
    dcm_dir = item_dict['dcm_dir']
    label_dir = item_dict['seg_label_dir']
    dcm_names = sorted(os.listdir(dcm_dir)) 
    dcm_list = []
    label_list = []
    # Read img and label files
    for dcm_name in dcm_names:
        # Image Dicom file
        dcm_file = pydicom.dcmread(os.path.join(dcm_dir, dcm_name),force=True)
        dcm = dcm_file.pixel_array
        # Label
        label_path = os.path.join(label_dir, dcm_name.replace('.dcm','.png'))
        label = cv2.imread(label_path, flags=0) if os.path.exists(label_path) else np.zeros_like(dcm)
        
        # Preprocessing the dicom pixel array
        rescale_intercept = float(dcm_file.RescaleIntercept)
        rescale_slope = float(dcm_file.RescaleSlope)
        dcm = dcm*rescale_slope+rescale_intercept
        Win_center, Win_width = args.dcm_win_center, args.dcm_win_width
        low_window = Win_center - (Win_width/2)
        high_window = Win_center + (Win_width/2)
        dcm = np.clip(dcm, low_window, high_window)
        dcm = (dcm-low_window)/(high_window-low_window)*255
        
        dcm_list.append(dcm)
        label_list.append(label)
        
    dcm = np.stack(dcm_list,axis=0).astype(np.float32)
    label = np.stack(label_list, axis=0).astype(np.float32)
    
    # unify the pixel spacing
    pixel_spacing = dcm_file.PixelSpacing[0]
    slice_thickness = item_dict['slice_thickness']
    D,H,W = dcm.shape
    dest_D = round(D*slice_thickness/pixel_spacing)
    # keep ratio rescaling
    dest_H,dest_W = args.rescale
    dest_D = round((dest_H/H)*dest_D)
    # Rescale
    dcm = torch.tensor(dcm).unsqueeze(0).unsqueeze(0)
    label = torch.tensor(label).unsqueeze(0).unsqueeze(0)
    dcm = F.interpolate(dcm, (dest_D, dest_H, dest_W), mode='trilinear').squeeze(0)/255
    label = F.interpolate(label, (dest_D, dest_H, dest_W), mode='nearest').squeeze(0)
    return dcm, label

def read_dcm_n_label_files_win(args, item_dict, mode, anno):
    dcm_dir = item_dict['dcm_dir']
    label_dir = item_dict['seg_label_dir']
    dcm_names = sorted(os.listdir(dcm_dir)) 
    dcm_list = []
    label_list = []
    # Read img and label files
    for dcm_name in dcm_names:
        # Image Dicom file
        dcm_file = pydicom.dcmread(os.path.join(dcm_dir, dcm_name),force=True)
        dcm = dcm_file.pixel_array
        # Label
        label_path = os.path.join(label_dir, dcm_name.replace('.dcm','.png'))
        label = cv2.imread(label_path, flags=0) if os.path.exists(label_path) else np.zeros_like(dcm)
        
        # Preprocessing the dicom pixel array
        rescale_intercept = float(dcm_file.RescaleIntercept)
        rescale_slope = float(dcm_file.RescaleSlope)
        dcm = dcm*rescale_slope+rescale_intercept
        Win_center, Win_width = args.dcm_win_center, args.dcm_win_width
        low_window = Win_center - (Win_width/2)
        high_window = Win_center + (Win_width/2)
        dcm = np.clip(dcm, low_window, high_window)
        dcm = (dcm-low_window)/(high_window-low_window)*255
        
        dcm_list.append(dcm)
        label_list.append(label)
        
    dcm = np.stack(dcm_list,axis=0).astype(np.float32)
    label = np.stack(label_list, axis=0).astype(np.float32)
    win_x1, win_y1, win_x2, win_y2 = anno['essential_bbox']
    dcm = dcm[:, win_y1:win_y2, win_x1:win_x2]
    label = label[:, win_y1:win_y2, win_x1:win_x2]
    
    # unify the pixel spacing
    D,H,W = dcm.shape
    pixel_spacing = dcm_file.PixelSpacing[0]
    slice_thickness = item_dict['slice_thickness']
    dest_D = round(D*slice_thickness/pixel_spacing)
    # keep ratio rescaling
    dest_H,dest_W = args.rescale
    dest_D = round((dest_H/H)*dest_D)
    ratio_D = dest_D/D
    # Rescale
    dcm = torch.tensor(dcm).unsqueeze(0).unsqueeze(0)
    label = torch.tensor(label).unsqueeze(0).unsqueeze(0)
    dcm = F.interpolate(dcm, (dest_D, dest_H, dest_W), mode='trilinear').squeeze(0)/255
    label = F.interpolate(label, (dest_D, dest_H, dest_W), mode='nearest').squeeze(0)
    # D_crop
    if args.D_center_ratio == 0:
        slice_start, slice_end = anno['essential_layers']['start'], anno['essential_layers']['stop']
        slice_center = round((slice_start+slice_end)/2)
        dest_slice_center = slice_center*ratio_D
        dest_slice_start = max(0, round(dest_slice_center - args.D_center_window/2))
        dest_slice_end = dest_slice_start + args.D_center_window
        dcm = dcm[dest_slice_start:dest_slice_end, :, :]
        label = label[dest_slice_start:dest_slice_end, :, :]
    
    return dcm, label

def read_img_n_label_dat_files(args, item_dict, mode, anno):
    img_dat_path = item_dict['image_dir'].replace('images','images_dat')+'.dat'
    # Read img and label files
    img_dict = pickle.load(open(img_dat_path, 'rb'))
    img = img_dict['img']
    label = img_dict['label']
    if args.use_detection:
        x1,y1,x2,y2 = anno['det_bbox']['x_min'], anno['det_bbox']['y_min'], anno['det_bbox']['x_max'], anno['det_bbox']['y_max']
        z_center_ratio = anno['det_bbox']['z_center_ratio']
        # H,W crop
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
        img = img[:, crp_y1:crp_y2, crp_x1:crp_x2]
        label = label[:, crp_y1:crp_y2, crp_x1:crp_x2]
    
    ori_D, ori_H, ori_W=img.shape
    rescale_ratio = args.rescale[0]/ori_H
    dest_D = int(ori_D * rescale_ratio)
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)/255
    label = torch.tensor(label).unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img,(dest_D, args.rescale[0], args.rescale[1]), mode='trilinear').squeeze(0)
    label = F.interpolate(label,(dest_D, args.rescale[0], args.rescale[1]), mode='nearest').squeeze(0)
    
    return img, label

def read_dcm_n_label_dat_files(args, item_dict, mode):
    dcm_dat_path = item_dict['dcm_dir'].replace('dcms','dcms_dat')+'.dat'
    # Read img and label files
    dcm_dict = pickle.load(open(dcm_dat_path, 'rb'))
    dcm = dcm_dict['dcm']
    label = dcm_dict['label']
    ori_D, ori_H, ori_W = dcm.shape
    rescale_ratio = args.rescale[0]/ori_H
    dest_D = round(ori_D * rescale_ratio)
    
    Win_center, Win_width = args.dcm_win_center, args.dcm_win_width
    low_window = Win_center - (Win_width/2)
    high_window = Win_center + (Win_width/2)
    dcm = np.clip(dcm, low_window, high_window)
    dcm = (dcm-low_window)/(high_window-low_window)
    
    dcm = torch.tensor(dcm).unsqueeze(0).unsqueeze(0)
    label = torch.tensor(label).unsqueeze(0).unsqueeze(0)
    dcm = F.interpolate(dcm,(dest_D, args.rescale[0], args.rescale[1]), mode='trilinear').squeeze(0)
    label = F.interpolate(label,(dest_D, args.rescale[0], args.rescale[1]), mode='nearest').squeeze(0)
    
    return dcm, label

def random_rotate(args, img, label):
    # Set random rotate and random translate
    rotate_degree = random.randint(args.rand_rotate[0], args.rand_rotate[1])
    # random rotation
    C,D,H,W = img.shape
    img = rotate(img, rotate_degree, InterpolationMode.BILINEAR)
    label = rotate(label, rotate_degree, InterpolationMode.NEAREST)
    return img, label

def random_scale(args, img, label):
    scale_factor = random.random()*(args.rand_scale[1]-args.rand_scale[0]) + args.rand_scale[0]
    img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode='trilinear').squeeze(0)
    label = F.interpolate(label.unsqueeze(0), scale_factor=scale_factor, mode='nearest').squeeze(0)
    return img, label

def random_crop(args, img, label, anno):
    C,D,H,W = img.shape
    # D crop
    if args.use_detection:
        center_slice = round(D*anno['det_bbox']['z_center_ratio'])
    else:
        center_slice = round(D*args.D_center_ratio)
    center_offset = (random.random()-0.5)*args.D_offset_ratio*D
    center_slice = center_slice+ center_offset
    center_window = args.D_center_window
    slice_interval = args.D_interval
    start_slice = max(0, round(center_slice-(center_window*slice_interval/2)))
    end_slice = min(D, start_slice + center_window*slice_interval)
    start_slice = end_slice - center_window*slice_interval
    # Random crop
    h_start = random.randrange(0, H-args.crop_size[1])
    w_start = random.randrange(0, W-args.crop_size[0])
    h_end = h_start + args.crop_size[1]
    w_end = w_start + args.crop_size[0]
    
    img = img[:, start_slice:end_slice:slice_interval, h_start:h_end, w_start:w_end]
    label = label[:, start_slice:end_slice:slice_interval, h_start:h_end, w_start:w_end]
    
    return img, label

def center_crop(args, img, label, anno):
    C,D,H,W = img.shape
    # D crop
    if args.use_detection:
        center_slice = round(D*anno['det_bbox']['z_center_ratio'])
    else:
        center_slice = round(D*args.D_center_ratio)
    center_window = args.D_center_window
    slice_interval = args.D_interval
    start_slice = max(0, round(center_slice-(center_window*slice_interval/2)))
    end_slice = min(D, start_slice + center_window*slice_interval)
    start_slice = end_slice - center_window*slice_interval
    # center crop
    h_start = round((H-args.crop_size[1])/2)
    w_start = round((W-args.crop_size[0])/2)
    h_end = h_start+args.crop_size[1]
    w_end = w_start+args.crop_size[0]
    img = img[:, start_slice:end_slice:slice_interval, h_start:h_end, w_start:w_end]
    label = label[:, start_slice:end_slice:slice_interval, h_start:h_end, w_start:w_end]
    
    return img, label

def random_crop_with_window(args, img, label):
    C,D,H,W = img.shape
    # D crop
    center_slice = round(D*args.D_center_ratio)
    center_window = args.D_center_window
    center_offset = (random.random()-0.5)*args.D_offset_ratio*args.D_center_window
    start_slice = max(0, round(center_slice+center_offset-center_window/2))
    end_slice = start_slice + args.D_center_window
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
    img = img[:, start_slice:end_slice, h_start:h_end, w_start:w_end]
    label = label[:, start_slice:end_slice, h_start:h_end, w_start:w_end]
    return img, label

def read_img_meta(anno):
    patient_id = anno['patient_id']
    essential_layers = anno['essential_layers']
    # Blood Measurement
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
    
    blood_input = torch.tensor([Hb, WBC, NEUT, PLT, ALB, ALT, AST, TB, DB, GGT, CEA, CA199],dtype=torch.float32)
    
    # other information
    if anno['sex']=='female':
        sex = 0
    elif anno['sex']=='male':
        sex = 1
    
    symptom = anno["symptom"]
    age = anno["age"]
    smoke = anno['smoke']
    diabetes = anno["diabetes"]
    heart_brain_blood_vessel = anno['heart_brain_blood_vessel']
    family = anno['family']
    others_input = torch.tensor([sex, symptom, age, smoke, diabetes, heart_brain_blood_vessel, family], dtype=torch.float32)
    
    blood_des = anno['blood_des']
    others_des = anno['others_des']
    
    img_meta = {
        'patient_id': patient_id,
        'essential_layers': essential_layers,
        "blood": blood_input, 
        "others": others_input,
        "blood_des": blood_des,
        "others_des": others_des,
        }
    return img_meta

def random_blood_text(blood_des):
    prefix, items_text = blood_des.split(':')
    items = items_text.split(';')
    item_num = len(items)
    items = random.sample(items, item_num)
    items_text = ';'.join(items)
    output_text = prefix+':'+ items_text+'.'
    return output_text

def random_others_text(others_des):
    output_text = others_des
    return output_text

def read_dcm_n_label_files_soft(args, item_dict, mode, anno):
    dcm_dir = item_dict['dcm_dir']
    label_dir = item_dict['seg_label_dir']
    dcm_names = sorted(os.listdir(dcm_dir)) 
    dcm_list = []
    label_list = []
    # Read img and label files
    for dcm_name in dcm_names:
        # Image Dicom file
        dcm_file = pydicom.dcmread(os.path.join(dcm_dir, dcm_name),force=True)
        dcm = dcm_file.pixel_array
        # Label
        label_path = os.path.join(label_dir, dcm_name.replace('.dcm','.png'))
        label = cv2.imread(label_path, flags=0) if os.path.exists(label_path) else np.zeros_like(dcm)
        
        # Preprocessing the dicom pixel array
        rescale_intercept = float(dcm_file.RescaleIntercept)
        rescale_slope = float(dcm_file.RescaleSlope)
        dcm = dcm*rescale_slope+rescale_intercept
        Win_center, Win_width = args.dcm_win_center, args.dcm_win_width
        low_window = Win_center - (Win_width/2)
        high_window = Win_center + (Win_width/2)
        dcm = np.clip(dcm, low_window, high_window)
        dcm = (dcm-low_window)/(high_window-low_window)*255
        
        dcm_list.append(dcm)
        label_list.append(label)
        
    dcm = np.stack(dcm_list,axis=0).astype(np.float32)
    label = np.stack(label_list, axis=0).astype(np.float32)
    win_x1, win_y1, win_x2, win_y2 = anno['essential_bbox']
    dcm = dcm[:, win_y1:win_y2, win_x1:win_x2]
    label = label[:, win_y1:win_y2, win_x1:win_x2]
    
    # unify the pixel spacing
    D,H,W = dcm.shape
    pixel_spacing = dcm_file.PixelSpacing[0]
    slice_thickness = item_dict['slice_thickness']
    dest_D = round(D*slice_thickness/pixel_spacing)
    # keep ratio rescaling
    dest_H,dest_W = args.rescale
    dest_D = round((dest_H/H)*dest_D)
    ratio_D = dest_D/D
    # Rescale
    dcm = torch.tensor(dcm).unsqueeze(0).unsqueeze(0)
    label = torch.tensor(label).unsqueeze(0).unsqueeze(0)
    dcm = F.interpolate(dcm, (dest_D, dest_H, dest_W), mode='trilinear').squeeze(0)/255
    label = F.interpolate(label, (dest_D, dest_H, dest_W), mode='nearest').squeeze(0)
    # D_crop
    esse_start, esse_end = anno['essential_layers']['start']*ratio_D, anno['essential_layers']['stop']*ratio_D
    info_dict = {
        'esse_start': esse_start,
        'esse_end': esse_end,
    }
    
    return dcm, label, info_dict

def random_crop_soft(args, img, label, info_dict):
    C,D,H,W = img.shape
    # D random crop
    slice_start = random.randint(round(D*0.1), round(D*0.9-args.D_center_window))
    slice_end = slice_start+args.D_center_window
    # Random crop
    h_start = random.randrange(0, H-args.crop_size[1])
    w_start = random.randrange(0, W-args.crop_size[0])
    h_end = h_start + args.crop_size[1]
    w_end = w_start + args.crop_size[0]
    img = img[:, slice_start:slice_end, h_start:h_end, w_start:w_end]
    label = label[:, slice_start:slice_end, h_start:h_end, w_start:w_end]
    
    esse_start, esse_end = info_dict['esse_start'], info_dict['esse_end']
    iou = cal_overlap(esse_start,esse_end, slice_start, slice_end)
    info_dict['slice_start'] = slice_start
    info_dict['slice_end'] = slice_end
    info_dict['iou'] = iou
    return img, label, info_dict

def center_crop_soft(args, img, label, info_dict):
    C,D,H,W = img.shape
    # center crop
    h_start = round((H-args.crop_size[1])/2)
    w_start = round((W-args.crop_size[0])/2)
    h_end = h_start+args.crop_size[1]
    w_end = w_start+args.crop_size[0]
    img = img[:, :, h_start:h_end, w_start:w_end]
    label = label[:, :, h_start:h_end, w_start:w_end]
    info_dict['iou'] = 1
    
    return img, label, info_dict
    
def cal_overlap(x_start,x_end,y_start,y_end):
    intersection = min(max(0, x_end-y_start), max(0,y_end-x_start))
    union = max(max(0, x_end-y_start), max(0,y_end-x_start))
    iou = intersection/union
    return iou