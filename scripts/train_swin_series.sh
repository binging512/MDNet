CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D/swinb_ep200_img_b4_rz256_crp224_z128.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D/swins_ep200_img_b8_rz256_crp224_z128.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D/swint_ep200_img_b8_rz256_crp224_z128.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D/swinb_ep200_img_b4_rz256_crp224_z64.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D/swins_ep200_img_b8_rz256_crp224_z64.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D/swint_ep200_img_b8_rz256_crp224_z64.yaml