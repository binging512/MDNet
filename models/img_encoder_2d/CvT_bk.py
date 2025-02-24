import sys
sys.path.append("/home/dmt218/zby/PANCLS")
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from models.img_encoder_2d.vit_pytorch.cvt import CvT


class CvT_Extractor(nn.Module):
    def __init__(self, args):
        super(CvT_Extractor, self).__init__()
        self.args = args
        assert args.net_seg_celoss == False, "Segmentation is not supported for CvT!"
        if args.net_backbone.lower() in ['cvt-13','cvt13']:
            s1_emb_dim, s2_emb_dim, s3_emb_dim = 64, 192, 384
            s1_heads, s2_heads, s3_heads = 1, 3, 6
            s1_depth, s2_depth, s3_depth = 1, 2, 10
        elif args.net_backbone.lower() in ['cvt-21','cvt21']:
            s1_emb_dim, s2_emb_dim, s3_emb_dim = 64, 192, 384
            s1_heads, s2_heads, s3_heads = 1, 3, 6
            s1_depth, s2_depth, s3_depth = 1, 4, 16
        elif args.net_backbone.lower() in ['cvt-w24','cvtw24']:
            s1_emb_dim, s2_emb_dim, s3_emb_dim = 192, 768, 1024
            s1_heads, s2_heads, s3_heads = 3, 12, 16
            s1_depth, s2_depth, s3_depth = 2, 2, 20
        else:
            raise NotImplementedError('Backbone {} is not implemented!!!'.format(args.net_backbone))
        
        self.cvt_model = CvT(
                            num_classes = 1000,
                            s1_emb_dim = s1_emb_dim, s1_emb_kernel = 7, s1_emb_stride = 4, s1_proj_kernel = 3, 
                            s1_kv_proj_stride = 2, s1_heads = s1_heads, s1_depth = s1_depth, s1_mlp_mult = 4,
                            s2_emb_dim = s2_emb_dim, s2_emb_kernel = 3, s2_emb_stride = 2, s2_proj_kernel = 3,
                            s2_kv_proj_stride = 2, s2_heads = s2_heads, s2_depth = s2_depth, s2_mlp_mult = 4,
                            s3_emb_dim = s3_emb_dim, s3_emb_kernel = 3, s3_emb_stride = 2, s3_proj_kernel = 3,
                            s3_kv_proj_stride = 2, s3_heads = s3_heads, s3_depth = s3_depth, s3_mlp_mult = 4,
                            dropout = 0.)
        
        self.proj = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(s3_emb_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.pretrained = nn.ModuleList([])
        self.new_added = nn.ModuleList([self.cvt_model, self.proj])
        
    def forward(self, x):
        pred, feat = self.cvt_model(x)
        feat = self.proj(feat).unsqueeze(1)
        pred_seg = torch.zeros_like(x)
        
        return feat, pred_seg    # BxLxC, BxCxHxW
    
if __name__=="__main__":
    model = CvT_Extractor(args=0).cuda()
    img = torch.zeros((1,3,512,512)).cuda()
    y = model(img)