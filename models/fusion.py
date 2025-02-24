import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, nhead: int, 
                 img_feat_channels = 512, blood_feat_channels=64, others_feat_channels=64,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AttentionFusion, self).__init__()
        self.norm_first = norm_first
        
        # img self-attention
        self.self_attn = nn.MultiheadAttention(img_feat_channels, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.img_norm = nn.LayerNorm(img_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.img_dropout = nn.Dropout(dropout)
        
        # img-blood attention
        self.blood_proj = nn.Linear(img_feat_channels, blood_feat_channels)
        self.blood_multihead_attn = nn.MultiheadAttention(blood_feat_channels, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.blood_norm1 = nn.LayerNorm(blood_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.blood_norm2 = nn.LayerNorm(blood_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.blood_dropout1 = nn.Dropout(dropout)
        self.blood_dropout2 = nn.Dropout(dropout)
        self.blood_linear1 = nn.Linear(blood_feat_channels, dim_feedforward, **factory_kwargs)
        self.blood_dropout = nn.Dropout(dropout)
        self.blood_linear2 = nn.Linear(dim_feedforward, blood_feat_channels, **factory_kwargs)
        
        # img-others attention
        self.others_proj = nn.Linear(img_feat_channels, others_feat_channels)
        self.others_multihead_attn = nn.MultiheadAttention(others_feat_channels, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.others_norm1 = nn.LayerNorm(others_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.others_norm2 = nn.LayerNorm(others_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.others_dropout1 = nn.Dropout(dropout)
        self.others_dropout2 = nn.Dropout(dropout)
        self.others_linear1 = nn.Linear(others_feat_channels, dim_feedforward, **factory_kwargs)
        self.others_dropout = nn.Dropout(dropout)
        self.others_linear2 = nn.Linear(dim_feedforward, others_feat_channels, **factory_kwargs)

        self.activation = activation


    def forward(self, img_feat, blood_feat, others_feat, 
                img_mask = None, blood_mask = None, others_mask = None,
                img_key_padding_mask = None, blood_key_padding_mask = None, others_key_padding_mask = None):
        x = img_feat
        if self.norm_first:
            # img self-attention
            x = self.img_norm(x)
            img_feat_attn = self.self_attn(x, x, x, 
                                            attn_mask=img_mask, 
                                            key_padding_mask=img_key_padding_mask, 
                                            need_weights=False)[0]
            img_feat_attn = x + self.img_dropout(img_feat_attn)
            # img-blood attention
            img_feat_attn1 = self.blood_proj(img_feat_attn)     # projection
            img_feat_attn1 = self.blood_norm1(img_feat_attn1)       # normalize
            img_blood_feat_attn = self.blood_multihead_attn(img_feat_attn1, blood_feat, blood_feat,
                                                            attn_mask=blood_mask,
                                                            key_padding_mask=blood_key_padding_mask,
                                                            need_weights=False)[0]                          # multihead attention
            img_blood_feat_attn = img_feat_attn1 + self.blood_dropout1(img_blood_feat_attn)
            img_blood_feat_attn = self.blood_norm2(img_blood_feat_attn)                                     # feedforward
            img_blood_feat_attn = self.blood_linear2(self.blood_dropout(self.activation(self.blood_linear1(img_blood_feat_attn))))
            img_blood_feat_attn = img_blood_feat_attn + self.blood_dropout2(img_blood_feat_attn)
        
            # img-others attention
            img_feat_attn2 = self.others_proj(img_feat_attn)
            img_feat_attn2 = self.others_norm1(img_feat_attn2)       # normalize
            img_others_feat_attn = self.others_multihead_attn(img_feat_attn2, others_feat, others_feat,
                                                                attn_mask=others_mask,
                                                                key_padding_mask=others_key_padding_mask,
                                                                need_weights=False)[0]                          # multihead attention
            img_others_feat_attn = img_feat_attn2 + self.others_dropout1(img_others_feat_attn)
            img_others_feat_attn = self.others_norm2(img_others_feat_attn)                                     # feedforward
            img_others_feat_attn = self.others_linear2(self.others_dropout(self.activation(self.others_linear1(img_others_feat_attn))))
            img_others_feat_attn = img_others_feat_attn + self.others_dropout2(img_others_feat_attn)
        else:
            # img self-attention
            img_feat_attn = self.self_attn(x, x, x, 
                                            attn_mask=img_mask, 
                                            key_padding_mask=img_key_padding_mask, 
                                            need_weights=False)[0]
            img_feat_attn = x + self.img_dropout(img_feat_attn)
            img_feat_attn = self.img_norm(img_feat_attn)
            # img-blood attention
            img_feat_attn1 = self.blood_proj(img_feat_attn)     # projection
            img_blood_feat_attn = self.blood_multihead_attn(img_feat_attn1, blood_feat, blood_feat,
                                                            attn_mask=blood_mask,
                                                            key_padding_mask=blood_key_padding_mask,
                                                            need_weights=False)[0]                          # multihead attention
            img_blood_feat_attn = blood_feat + self.blood_dropout1(img_blood_feat_attn)
            img_blood_feat_attn = self.blood_norm1(img_blood_feat_attn)       # normalize
            
            img_blood_feat_attn = self.blood_linear2(self.blood_dropout(self.activation(self.blood_linear1(img_blood_feat_attn))))  # feedforward
            img_blood_feat_attn = img_blood_feat_attn + self.blood_dropout2(img_blood_feat_attn)
            img_blood_feat_attn = self.blood_norm2(img_blood_feat_attn)
        
            # img-others attention
            img_feat_attn2 = self.others_proj(img_feat_attn)
            img_others_feat_attn = self.others_multihead_attn(img_feat_attn2, others_feat, others_feat,
                                                                attn_mask=others_mask,
                                                                key_padding_mask=others_key_padding_mask,
                                                                need_weights=False)[0]                          # multihead attention
            img_others_feat_attn = others_feat + self.others_dropout1(img_others_feat_attn)
            img_others_feat_attn = self.others_norm1(img_others_feat_attn)       # normalize
            
            img_others_feat_attn = self.others_linear2(self.others_dropout(self.activation(self.others_linear1(img_others_feat_attn)))) # feedforward
            img_others_feat_attn = img_others_feat_attn + self.others_dropout2(img_others_feat_attn)
            img_others_feat_attn = self.others_norm2(img_others_feat_attn)
        
        # Concatenate
        img_feat = torch.mean(img_feat,dim=1)
        img_blood_feat_attn = torch.mean(img_blood_feat_attn, dim=1)
        img_others_feat_attn = torch.mean(img_others_feat_attn, dim=1)
        fused_feat = torch.concat((img_feat, img_blood_feat_attn, img_others_feat_attn), dim=1)
        return fused_feat


class MLP_Mixer_C_only(nn.Module):
    def __init__(self, in_channels, expansion_factor = 4, dropout = 0) -> None:
        super(MLP_Mixer_C_only, self).__init__()
        
        self.norm = nn.LayerNorm(in_channels)
        self.fn = nn.Sequential(
            nn.Linear(in_channels, round(in_channels*expansion_factor)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(round(in_channels*expansion_factor), in_channels),
            nn.Dropout(dropout),
        )
        
    def forward(self, img_feat, blood_feat, others_feat):
        x = torch.cat((img_feat, blood_feat, others_feat), dim=2)
        x = x.squeeze(1)
        x = self.fn(self.norm(x)) + x
        return x
    

class MLP_Mixer(nn.Module):
    def __init__(self, token_dims, channel_dims, token_expansion=0.5, channel_expansion=4, dropout = 0) -> None:
        super(MLP_Mixer, self).__init__()
        # Token
        self.norm_t = nn.LayerNorm(channel_dims)
        self.fn_t = nn.Sequential(
            nn.Conv1d(token_dims, round(token_dims*token_expansion), 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(round(token_dims*token_expansion), token_dims, 1),
            nn.Dropout(dropout),
        )
        
        # Channel
        self.norm_c = nn.LayerNorm(channel_dims)
        self.fn_c = nn.Sequential(
            nn.Linear(channel_dims, round(channel_dims*channel_expansion)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(round(channel_dims*channel_expansion), channel_dims),
            nn.Dropout(dropout),
        )
        
    def forward(self, img_feat, blood_feat, others_feat):
        B,L,C = img_feat.shape
        if blood_feat.shape[1] == 1:
            blood_feat = blood_feat.repeat(1, L, 1)
            others_feat = blood_feat.repeat(1, L, 1)
        
        x = torch.cat((img_feat, blood_feat, others_feat), dim=2)
        x = self.fn_t(self.norm_t(x)) + x
        x = self.fn_c(self.norm_c(x)) + x
        x = torch.mean(x, dim=1)
        return x


class Attention_Mixer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, nhead: int, 
                 img_feat_channels = 512, blood_feat_channels=64, others_feat_channels=64,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 token_dims= 257, channel_dims= 640, token_expansion=0.5, channel_expansion=4, mixer_dropout = 0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Attention_Mixer, self).__init__()
        self.norm_first = norm_first
        
        # img self-attention
        self.self_attn = nn.MultiheadAttention(img_feat_channels, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.img_norm = nn.LayerNorm(img_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.img_dropout = nn.Dropout(dropout)
        
        # img-blood attention
        self.blood_proj = nn.Linear(img_feat_channels, blood_feat_channels)
        self.blood_multihead_attn = nn.MultiheadAttention(blood_feat_channels, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.blood_norm1 = nn.LayerNorm(blood_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.blood_norm2 = nn.LayerNorm(blood_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.blood_dropout1 = nn.Dropout(dropout)
        self.blood_dropout2 = nn.Dropout(dropout)
        self.blood_linear1 = nn.Linear(blood_feat_channels, dim_feedforward, **factory_kwargs)
        self.blood_dropout = nn.Dropout(dropout)
        self.blood_linear2 = nn.Linear(dim_feedforward, blood_feat_channels, **factory_kwargs)
        
        # img-others attention
        self.others_proj = nn.Linear(img_feat_channels, others_feat_channels)
        self.others_multihead_attn = nn.MultiheadAttention(others_feat_channels, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.others_norm1 = nn.LayerNorm(others_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.others_norm2 = nn.LayerNorm(others_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.others_dropout1 = nn.Dropout(dropout)
        self.others_dropout2 = nn.Dropout(dropout)
        self.others_linear1 = nn.Linear(others_feat_channels, dim_feedforward, **factory_kwargs)
        self.others_dropout = nn.Dropout(dropout)
        self.others_linear2 = nn.Linear(dim_feedforward, others_feat_channels, **factory_kwargs)

        self.activation = activation

        self.mixer = MLP_Mixer(token_dims=token_dims, channel_dims=channel_dims, 
                               token_expansion=token_expansion, channel_expansion=channel_expansion,
                               dropout=mixer_dropout)

    def forward(self, img_feat, blood_feat, others_feat, 
                img_mask = None, blood_mask = None, others_mask = None,
                img_key_padding_mask = None, blood_key_padding_mask = None, others_key_padding_mask = None):
        x = img_feat
        if self.norm_first:
            # img self-attention
            x = self.img_norm(x)
            img_feat_attn = self.self_attn(x, x, x, 
                                            attn_mask=img_mask, 
                                            key_padding_mask=img_key_padding_mask, 
                                            need_weights=False)[0]
            img_feat_attn = x + self.img_dropout(img_feat_attn)
            # img-blood attention
            img_feat_attn1 = self.blood_proj(img_feat_attn)     # projection
            img_feat_attn1 = self.blood_norm1(img_feat_attn1)       # normalize
            img_blood_feat_attn = self.blood_multihead_attn(img_feat_attn1, blood_feat, blood_feat,
                                                            attn_mask=blood_mask,
                                                            key_padding_mask=blood_key_padding_mask,
                                                            need_weights=False)[0]                          # multihead attention
            img_blood_feat_attn = img_feat_attn1 + self.blood_dropout1(img_blood_feat_attn)
            img_blood_feat_attn = self.blood_norm2(img_blood_feat_attn)                                     # feedforward
            img_blood_feat_attn = self.blood_linear2(self.blood_dropout(self.activation(self.blood_linear1(img_blood_feat_attn))))
            img_blood_feat_attn = img_blood_feat_attn + self.blood_dropout2(img_blood_feat_attn)
        
            # img-others attention
            img_feat_attn2 = self.others_proj(img_feat_attn)
            img_feat_attn2 = self.others_norm1(img_feat_attn2)       # normalize
            img_others_feat_attn = self.others_multihead_attn(img_feat_attn2, others_feat, others_feat,
                                                                attn_mask=others_mask,
                                                                key_padding_mask=others_key_padding_mask,
                                                                need_weights=False)[0]                          # multihead attention
            img_others_feat_attn = img_feat_attn2 + self.others_dropout1(img_others_feat_attn)
            img_others_feat_attn = self.others_norm2(img_others_feat_attn)                                     # feedforward
            img_others_feat_attn = self.others_linear2(self.others_dropout(self.activation(self.others_linear1(img_others_feat_attn))))
            img_others_feat_attn = img_others_feat_attn + self.others_dropout2(img_others_feat_attn)
        else:
            # img self-attention
            img_feat_attn = self.self_attn(x, x, x, 
                                            attn_mask=img_mask, 
                                            key_padding_mask=img_key_padding_mask, 
                                            need_weights=False)[0]
            img_feat_attn = x + self.img_dropout(img_feat_attn)
            img_feat_attn = self.img_norm(img_feat_attn)
            # img-blood attention
            img_feat_attn1 = self.blood_proj(img_feat_attn)     # projection
            img_blood_feat_attn = self.blood_multihead_attn(img_feat_attn1, blood_feat, blood_feat,
                                                            attn_mask=blood_mask,
                                                            key_padding_mask=blood_key_padding_mask,
                                                            need_weights=False)[0]                          # multihead attention
            img_blood_feat_attn = blood_feat + self.blood_dropout1(img_blood_feat_attn)
            img_blood_feat_attn = self.blood_norm1(img_blood_feat_attn)       # normalize
            
            img_blood_feat_attn = self.blood_linear2(self.blood_dropout(self.activation(self.blood_linear1(img_blood_feat_attn))))  # feedforward
            img_blood_feat_attn = img_blood_feat_attn + self.blood_dropout2(img_blood_feat_attn)
            img_blood_feat_attn = self.blood_norm2(img_blood_feat_attn)
        
            # img-others attention
            img_feat_attn2 = self.others_proj(img_feat_attn)
            img_others_feat_attn = self.others_multihead_attn(img_feat_attn2, others_feat, others_feat,
                                                                attn_mask=others_mask,
                                                                key_padding_mask=others_key_padding_mask,
                                                                need_weights=False)[0]                          # multihead attention
            img_others_feat_attn = others_feat + self.others_dropout1(img_others_feat_attn)
            img_others_feat_attn = self.others_norm1(img_others_feat_attn)       # normalize
            
            img_others_feat_attn = self.others_linear2(self.others_dropout(self.activation(self.others_linear1(img_others_feat_attn)))) # feedforward
            img_others_feat_attn = img_others_feat_attn + self.others_dropout2(img_others_feat_attn)
            img_others_feat_attn = self.others_norm2(img_others_feat_attn)
        
        # Concatenate
        fused_feat = self.mixer(img_feat, img_blood_feat_attn, img_others_feat_attn)
        
        return fused_feat
    
    
class AttentionImgFusion(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, nhead: int, 
                 img_feat_channels = 512, blood_feat_channels=64, others_feat_channels=64,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AttentionImgFusion, self).__init__()
        self.norm_first = norm_first
        
        # img self-attention
        self.self_attn = nn.MultiheadAttention(img_feat_channels, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.img_norm = nn.LayerNorm(img_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.img_dropout = nn.Dropout(dropout)
        
        # img-blood attention
        self.blood_proj = nn.Linear(img_feat_channels, blood_feat_channels)
        self.blood_multihead_attn = nn.MultiheadAttention(blood_feat_channels, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.blood_norm1 = nn.LayerNorm(blood_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.blood_norm2 = nn.LayerNorm(blood_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.blood_dropout1 = nn.Dropout(dropout)
        self.blood_dropout2 = nn.Dropout(dropout)
        self.blood_linear1 = nn.Linear(blood_feat_channels, dim_feedforward, **factory_kwargs)
        self.blood_dropout = nn.Dropout(dropout)
        self.blood_linear2 = nn.Linear(dim_feedforward, blood_feat_channels, **factory_kwargs)
        
        # img-others attention
        self.others_proj = nn.Linear(img_feat_channels, others_feat_channels)
        self.others_multihead_attn = nn.MultiheadAttention(others_feat_channels, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.others_norm1 = nn.LayerNorm(others_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.others_norm2 = nn.LayerNorm(others_feat_channels, eps=layer_norm_eps, **factory_kwargs)
        self.others_dropout1 = nn.Dropout(dropout)
        self.others_dropout2 = nn.Dropout(dropout)
        self.others_linear1 = nn.Linear(others_feat_channels, dim_feedforward, **factory_kwargs)
        self.others_dropout = nn.Dropout(dropout)
        self.others_linear2 = nn.Linear(dim_feedforward, others_feat_channels, **factory_kwargs)

        self.activation = activation


    def forward(self, img_feat, blood_feat, others_feat, 
                img_mask = None, blood_mask = None, others_mask = None,
                img_key_padding_mask = None, blood_key_padding_mask = None, others_key_padding_mask = None):
        x = img_feat
        # img self-attention
        img_feat_attn = self.self_attn(x, x, x, 
                                        attn_mask=img_mask, 
                                        key_padding_mask=img_key_padding_mask, 
                                        need_weights=False)[0]
        img_feat_attn = x + self.img_dropout(img_feat_attn)
        img_feat_attn = self.img_norm(img_feat_attn)
        # img-blood attention
        img_feat_attn1 = self.blood_proj(img_feat_attn)     # projection
        img_blood_feat_attn = self.blood_multihead_attn(blood_feat, img_feat_attn1, img_feat_attn1,
                                                        attn_mask=blood_mask,
                                                        key_padding_mask=blood_key_padding_mask,
                                                        need_weights=False)[0]                          # multihead attention
        img_blood_feat_attn = img_feat_attn1 + self.blood_dropout1(img_blood_feat_attn)
        img_blood_feat_attn = self.blood_norm1(img_blood_feat_attn)       # normalize
        
        img_blood_feat_attn = self.blood_linear2(self.blood_dropout(self.activation(self.blood_linear1(img_blood_feat_attn))))  # feedforward
        img_blood_feat_attn = img_blood_feat_attn + self.blood_dropout2(img_blood_feat_attn)
        img_blood_feat_attn = self.blood_norm2(img_blood_feat_attn)
    
        # img-others attention
        img_feat_attn2 = self.others_proj(img_feat_attn)
        img_others_feat_attn = self.others_multihead_attn(others_feat, img_feat_attn2, img_feat_attn2,
                                                            attn_mask=others_mask,
                                                            key_padding_mask=others_key_padding_mask,
                                                            need_weights=False)[0]                          # multihead attention
        img_others_feat_attn = img_feat_attn2 + self.others_dropout1(img_others_feat_attn)
        img_others_feat_attn = self.others_norm1(img_others_feat_attn)       # normalize
        
        img_others_feat_attn = self.others_linear2(self.others_dropout(self.activation(self.others_linear1(img_others_feat_attn)))) # feedforward
        img_others_feat_attn = img_others_feat_attn + self.others_dropout2(img_others_feat_attn)
        img_others_feat_attn = self.others_norm2(img_others_feat_attn)
        
        # Concatenate
        img_feat = torch.mean(img_feat,dim=1)
        img_blood_feat_attn = torch.mean(img_blood_feat_attn, dim=1)
        img_others_feat_attn = torch.mean(img_others_feat_attn, dim=1)
        fused_feat = torch.concat((img_feat, img_blood_feat_attn, img_others_feat_attn), dim=1)
        return fused_feat
    
    
class MLP_fusion(nn.Module):
    def __init__(self, img_feat_channels = 512, blood_feat_channels=64, others_feat_channels=64,) -> None:
        super(MLP_fusion, self).__init__()
        self.blood_proj = nn.Sequential(
            nn.Linear(img_feat_channels, blood_feat_channels),
            nn.LayerNorm(blood_feat_channels),
            nn.ReLU()
        )
        self.others_proj = nn.Sequential(
            nn.Linear(img_feat_channels, others_feat_channels),
            nn.LayerNorm(blood_feat_channels),
            nn.ReLU()
        )
        
    def forward(self,img_feat, blood_feat, others_feat):
        img_feat = torch.mean(img_feat, dim=1)
        blood_feat = torch.mean(blood_feat, dim=1)
        others_feat = torch.mean(others_feat, dim=1)
        
        img_blood_feat = self.blood_proj(img_feat)
        img_blood_feat = blood_feat + img_blood_feat*blood_feat
        img_others_feat = self.others_proj(img_feat)
        img_others_feat = others_feat + img_others_feat* others_feat
        fused_feat = torch.cat((img_feat, blood_feat, others_feat), dim=1)
        return fused_feat