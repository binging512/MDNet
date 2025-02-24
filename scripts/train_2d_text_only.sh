CUDA_VISIBLE_DEVICES=0,1 python run.py --train_2d_pass True --config configs/2D_paper/none_mlp_none_cat_iter500_img_b16_rz560_crp512_ecs_score.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_2d_pass True --config configs/2D_paper/none_none_trans_cat_iter500_img_b16_rz560_crp512_ecs_score.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_2d_pass True --config configs/2D_paper/none_mlp_mlp_cat_iter500_img_b16_rz560_crp512_ecs_score.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_2d_pass True --config configs/2D_paper/none_mlp_trans_cat_iter500_img_b16_rz560_crp512_ecs_score.yaml
CUDA_VISIBLE_DEVICES=0,1 python run.py --train_2d_pass True --config configs/2D_paper/none_trans_trans_cat_iter500_img_b16_rz560_crp512_ecs_score.yaml
