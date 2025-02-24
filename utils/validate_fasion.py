import numpy as np
import torch

def slide_window(args, model, input_batch):
    whole_img = input_batch['img']
    B,C,D,H,W = whole_img.shape
    d_stride = args.slide_stride
    d_size = args.D_center_window
    d_grids = max(D - d_size + d_stride - 1, 0) // d_stride + 1
    output_dict = {
        'pred_invade': [],
        'pred_surgery': [],
        'ct_pred_invade': [],
        'ct_pred_surgery': [],
        'pred_invade_cls': [],
        'pred_surgery_cls': [],
        'ct_pred_invade_cls': [],
        'ct_pred_surgery_cls': []
    }
    for d_idx in range(d_grids):
        d1 = d_idx * d_stride
        d2 = min(d1 + d_size, D)
        d1 = max(d2 - d_size, 0)
        d_crop_img = whole_img[:, :, d1:d2, :, :]
        crop_input_batch = {
            'img': d_crop_img,
            "blood": input_batch['blood'],
            'others': input_batch['others'],
            "blood_des_1": input_batch['blood_des_1'],
            "blood_des_2": input_batch['blood_des_2'],
            "blood_des": input_batch['blood_des'],
            "others_des": input_batch['others_des'],
        }
        crop_output_dict = one_piece(args, model, crop_input_batch)
        output_dict['pred_invade'].append(crop_output_dict['pred_invade'])
        output_dict['pred_surgery'].append(crop_output_dict['pred_surgery'])
        output_dict['ct_pred_invade'].append(crop_output_dict['ct_pred_invade'])
        output_dict['ct_pred_surgery'].append(crop_output_dict['ct_pred_surgery'])
        output_dict['pred_invade_cls'].append(crop_output_dict['pred_invade_cls'])
        output_dict['pred_surgery_cls'].append(crop_output_dict['pred_surgery_cls'])
        output_dict['ct_pred_invade_cls'].append(crop_output_dict['ct_pred_invade_cls'])
        output_dict['ct_pred_surgery_cls'].append(crop_output_dict['ct_pred_surgery_cls'])
    
    output_dict['pred_invade'],_ = torch.max(torch.stack(output_dict['pred_invade'], dim=-1), dim=-1)
    output_dict['pred_surgery'],_ = torch.max(torch.stack(output_dict['pred_surgery'], dim=-1), dim=-1)
    output_dict['ct_pred_invade'],_ = torch.max(torch.stack(output_dict['ct_pred_invade'], dim=-1), dim=-1)
    output_dict['ct_pred_surgery'],_ = torch.max(torch.stack(output_dict['ct_pred_surgery'], dim=-1), dim=-1)

    output_dict['pred_invade_cls'] = np.max(np.stack(output_dict['pred_invade_cls'],axis=0),axis=0)
    output_dict['pred_surgery_cls'] = np.max(np.stack(output_dict['pred_surgery_cls'],axis=0),axis=0)
    output_dict['ct_pred_invade_cls'] = np.max(np.stack(output_dict['ct_pred_invade_cls'],axis=0),axis=0)
    output_dict['ct_pred_surgery_cls'] = np.max(np.stack(output_dict['ct_pred_surgery_cls'],axis=0),axis=0)
    
    
    return output_dict

def one_piece(args, model, input_batch):
    with torch.no_grad():
        output = model(input_batch)
        preds_invade = output['preds_invade']
        preds_surgery = output['preds_surgery']
        ct_preds_invade = output['ct_preds_invade']
        ct_preds_surgery = output['ct_preds_surgery']
            
    # Multi-Head Fusion
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
    
    pred_invade_cls = torch.argmax(pred_invade.detach().cpu(),dim=1).numpy().astype(np.int32)
    pred_surgery_cls = torch.argmax(pred_surgery.detach().cpu(),dim=1).numpy().astype(np.int32)
    ct_pred_invade_cls = torch.argmax(ct_pred_invade.detach().cpu(),dim=1).numpy().astype(np.int32)
    ct_pred_surgery_cls = torch.argmax(ct_pred_surgery.detach().cpu(),dim=1).numpy().astype(np.int32)
    
    output_dict = {
        'pred_invade': pred_invade,
        'pred_surgery': pred_surgery,
        'ct_pred_invade': ct_pred_invade,
        'ct_pred_surgery': ct_pred_surgery,
        'pred_invade_cls': pred_invade_cls,
        'pred_surgery_cls': pred_surgery_cls,
        'ct_pred_invade_cls': ct_pred_invade_cls,
        'ct_pred_surgery_cls': ct_pred_surgery_cls
    }
    return output_dict