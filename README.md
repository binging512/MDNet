# MDNet

## Introduction

This is the official implement of paper: TBD

Highlights:

- A multi-modal pancreatic cancer dataset (MPCD) is constructed to serve for the identification of SMPV invasion and treatment recommendation. MPCD covers preoperative CT images, blood examination results, personal data, and postoperative pathological data of 208 pancreatic cancer patients. To our knowledge, this is first multi-modal pancreatic cancer dataset.

- We propose a new multi-modal prediction baseline model. With the effective representation of three-modal data, the alignment and fusion of features, it jointly accomplishes two tasks, namely the diagnosis of SMPV invasion and treatment recommendation. 

- The experimental results on the MPCD demonstrate that the effectiveness of our method for non-invasive SMPV diagnosis as well as the potential for personalized treatment recommendation. The best performance also shows the comprehensiveness and rationality of the proposed multi-modal approach compared with the existing single-modal methods.

![](.\figs\framework.png)

## MPCD

The multi-modal pancreatic cancer dataset (MPCD) contains 208 pancreatic cancer patients, which covers preoperative CT images, blood examination results, personal data, and postoperative pathological data of each patient.

Download Link: [BaiduDiskLink](https://pan.baidu.com/s/1sPmA0VMD5Wu287yYNiFT5A?pwd=5hmd)

## Instruction

### Installation

1. Download the code

```shell
git clone https://github.com/binging512/MDNet.git
cd MDNet
```

2. Create the environment

```shell
conda create -n mdnet python=3.9
```

3. Install the requirements

```shell
pip install -r requirements.txt
```

### Prepare the datasets

1. Download the MPCD from [].

2. Place the dataset into ```./data/MPCD``` directory.

3. Refer the [deformableLKA](https://github.com/xmindflow/deformableLKA) and save the coarse detection results into ```data/MPCD/det_bbox.json```.

   Or you can directly use the existing detection results.

4. Please make sure the split file contains the correct path information of the dataset.

   split_2d.json :  (For 2d-based prediction)

```json
{
  "train": [																	# Training items
    {
      "anno_item": "0000000136",												# Case ID
      "image_path": "data/MPCD/images/0000000136/ser009img00001.jpg",			# Image path (.jpg files)
      "dcm_path": "data/MPCD/dcms/0000000136/ser009img00001.dcm",				# Dicom path (.dcm files)
      "seg_label_path": "data/MPCD/seg_labels/0000000136/ser009img00001.png",	# Tumor and vein segmentation labels (.png files)
      "prev_image_path": "data/MPCD/images/0000000136/ser009img00001.jpg",		# Previous image path (.jpg files)
      "next_image_path": "data/MPCD/images/0000000136/ser009img00002.jpg",		# Next image path (.jpg files)
      "prev_dcm_path": "data/MPCD/dcms/0000000136/ser009img00001.dcm",			# Previous dicom path (.dcm files)
      "next_dcm_path": "data/MPCD/dcms/0000000136/ser009img00002.dcm"			# Next dicom path (.dcm files)
    },
    ...
  ],
  "val": [...],																	# Validating items
  "test": [...],																# Testing items
}
```

​		split_3d.json:  (For 3d-based prediction)

```json
{
  "train": [																# Training items
    {
      "anno_item": "0000000136",											# Case ID
      "image_dir": "data/MPCD/images/0000000136",							# Image directory (.jpg files)
      "dcm_dir": "data/MPCD/dcms/0000000136",								# Dicom directory (.dcm files)
      "seg_label_dir": "data/MPCD/seg_labels/0000000136",					# Tumor and vein segmentation labels (.png files)
      "slice_thickness": 1.0,
      "essential_layers": {													# The slices contain the tumor
        "start": 84,
        "stop": 160
      }
    },
    ...
   ],
  "val": [...],																# Validating items
  "test": [...],															# Testing items
}
```

### Config

Before training, please customized the config files in `./configs`. Here is the detailed description for the config (an example of `configs/2D_paper_final/res50_mlp_trans_attn_iter5000_img_b16_rz512_win360_crp256_ecs_score.yaml`):

```yaml
# Dataset
data_type: "Pancreas_2d"							# "Pancreas_2d" or "Pancreas_3d" for 2d-based or 3d-based model
data_mode: 'img'									# "img" or "dcm" for using .jpg or .dcm files as the image inputs
train_mode: 'all'									# "train" or 'all' for using training cases or training+val cases
test_mode: 'val'
data_root: './data'									# Data root
split_info: './data/MPCD/splits/split_2d.json'		# Split file path
cls_anno_path: './data/MPCD/anno_all.json'          # Annotation file path

batch_size: 16										# Training and testing batch size
num_worker: 8										# Number workers for reading
drop_last: True

# Dicom setting										# Only used when "data_mode" is "dcm"
dcm_win_center: 40									# Win_center for dicom files
dcm_win_width: 350									# Win_width for dicom files

# Augmentation										# Data augmentation 
rescale: [360,360]
rand_rotate: ""
rand_scale: ""
crop_window: [360, 360]
crop_size: [256,256]
rand_blood_text: False
rand_others_text: False
window_crop: False

## 3D
D_center_ratio: 0.5
D_offset_ratio: 0.1
D_center_window: 64
use_detection: True									# Whether use deformableLKA detection results for 3d-based model
## 2D
neighbour_slice: True
use_detection_slice: True							# Whether use deformableLKA detection results for 2d-based model

# Net												# Network setting
net_resume: ''
# Image Encoder
net_name: 'resnet'									# Image encoder name
net_backbone: 'resnet50'
net_pretrain: ''
# Text Encoder
net_blood_name: "mlp"								# Blood encoder ("none", "mlp" and "transformer" for no data, tensorized data, text data)
net_others_name: "transformer"						# Person encoder ("none", "mlp" and "transformer" for no data, tensorized data, text data)
net_text_pretrain: 'CLIP_ViT-B/32' 					# Text encoder ('CLIP_ViT-B/32', 'CLIP_ViT-L/14')
# Feature Fusion
net_fusion: 'attn'									# 'concat', 'mlp', 'attn' for the three proposed fusion modules respectively
# Classifiers										# Classifier setting
net_dropout: 0.5
net_classifier_inchannel: 640
net_essential_classes: 2
net_essential_object: 'tumor'
net_invade_classes: 2
net_invade_weights: [1, 1]
net_surgery_classes: 2
net_seg_classes: 3
net_seg_weights: [1.,10.,10.]
net_nheads: 1
net_num_iter: 5000
net_learning_rate: 0.001
net_cls_use_essential: True							# Whether the classification loss will be masked by the key slice label
net_seg_use_essential: True							# Whether the segmentation loss will be masked by the key slice label
net_invade_cls_celoss: True							# SMPV invasion celoss
net_surgery_cls_celoss: True						# Treatment recommendation celoss
net_essential_cls_celoss: True						# Key slice prediction celoss
net_seg_celoss: False								# Tumor and vein segmentation celoss

# Inference
test_fusion: 'mean'									
main_metric: 'test_avg'
test_checkpoint: 'checkpoints/iter_3900.pth'		# Testing trained model file path
thres_invade: 0.5
thres_surgery: 0.5

## 2D
vote_mode: 'score_essential'						# The Topk voting with key slice scores
score_mode: 'max'

# Saving
workspace: './workspace/2D_paper_final/res50_mlp_trans_attn_iter5000_img_b16_det_rz360_crp256_ecs_score'
results_val: 'results_val'
results_test: 'results_test'
checkpoint: best_model.pth
val_interval: 100
save_interval: 100
```

### Training and testing

To train the model, please refer to the following command:

- 2d-based model training:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --train_2d_pass True --config <path/to/config>
```

- 3d-based model training:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --train_3d_pass True --config <path/to/config>
```

And, to test the model, please refer to the following command:

- 2d-based model testing:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --test_2d_pass True --config <path/to/config>
```

- 3d-based model testing:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --test_3d_pass True --config <path/to/config>
```

## Acknowledgement

Great thanks to [deformableLKA](https://github.com/xmindflow/deformableLKA)!

## Citation

```
[TBD]
```



