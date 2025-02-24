import torch
import torch.nn as nn
from models.img_encoder_2d.vit_pytorch import ViT

class ViT_Extractor(nn.Module):
    def __init__(self, args):
        super(ViT_Extractor, self).__init__()
        self.args = args
        assert args.net_seg_celoss == False, "Segmentation is not supported for ViT!"
        self.vit_model = ViT(image_size = args.crop_size[0],
                                patch_size = 32,
                                num_classes = 1000,
                                dim = 1024,
                                depth = 24,
                                heads = 16,
                                mlp_dim = 4096,
                                dropout = 0.1,
                                emb_dropout = 0.1, 
                                pool = 'mean')
        self.proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.pretrained = nn.ModuleList([])
        self.new_added = nn.ModuleList([self.vit_model, self.proj])
        
    def forward(self, x):
        pred, feat = self.vit_model(x)
        feat = torch.mean(feat,dim=1,keepdim=True)
        feat = self.proj(feat)
        
        pred_seg = torch.zeros_like(x)
        
        return feat, pred_seg    # BxLxC, BxCxHxW
    
if __name__=="__main__":
    model = ViT_Extractor(args=0)
    img = torch.zeros((1,3,256,256))
    y = model(img)
    