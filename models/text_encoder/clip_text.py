import torch
import torch.nn as nn
import torch.nn.functional as F
import models.text_encoder.clip as clip

class CLIP_Encoder(nn.Module):
    def __init__(self, clip_name, outchannels=256, context_length = 77, pretrain= True) -> None:
        super(CLIP_Encoder, self).__init__()
        
        self.clip, self.clip_preprocess = clip.load(clip_name.split('_')[1], context_length=context_length)
        if pretrain==True:
            for key, value in self.clip.named_parameters():
                if key == 'positional_embedding':
                    value.requires_grad = True
                else:
                    value.requires_grad = False
            pass
        else:
            self.clip.initialize_parameters()
            for key, value in self.clip.named_parameters():
                if 'visual' in key:
                    value.requires_grad=False
                else:
                    value.requires_grad=True
            
        if clip_name=='CLIP_ViT-B/32':
            text_dim = 512
        elif clip_name=='CLIP_ViT-L/14':
            text_dim = 768
        self.mlp_text = nn.Sequential(nn.Linear(text_dim, text_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(text_dim, outchannels))

    def forward(self, x):
        features_text = self.clip.encode_text(x)
        features_text = features_text.to(torch.float)
        features_text = self.mlp_text(features_text)
        features_text = F.normalize(features_text, p=2, dim=-1).unsqueeze(1)
        return features_text