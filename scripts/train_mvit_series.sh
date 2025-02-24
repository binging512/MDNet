CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D/mvit_ep200_img_b16_rz256_crp224_z128.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D/mvit_ep300_img_b16_rz256_crp224_z128.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D/mvit_ep400_img_b16_rz256_crp224_z128.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D/mvit_ep200_img_b16_rz256_crp224_z64.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D/mvit_ep300_img_b16_rz256_crp224_z64.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D/mvit_ep400_img_b16_rz256_crp224_z64.yaml
