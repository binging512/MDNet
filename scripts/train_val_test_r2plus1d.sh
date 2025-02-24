CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/r2plus1d18_mlp_mlp_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/r2plus1d18_mlp_trans_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/r2plus1d18_trans_trans_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/r2plus1d18c_mlp_mlp_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/r2plus1d18c_mlp_trans_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_3d_pass True --config configs/3D_fuse_val_test/r2plus1d18c_trans_trans_ep100_img_b8_rz360_win256_crp224_z64_0.4.yaml