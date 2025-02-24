CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/swinb_mlp_mlp_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/swinb_mlp_trans_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/swinb_trans_trans_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/swinbc_mlp_mlp_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/swinbc_mlp_trans_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=2,3 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/swinbc_trans_trans_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml