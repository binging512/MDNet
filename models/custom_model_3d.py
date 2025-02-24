import torch
import torch.nn as nn
import torch.nn.functional as F
from .img_encoder import *
from .text_encoder import *
from .img_encoder.R2plus1d import SpatioTemporalResBlock
from .fusion import AttentionFusion, MLP_Mixer, MLP_Mixer_C_only, Attention_Mixer, MLP_fusion, AttentionImgFusion

class MMFF(nn.Module):
    def __init__(self, args) -> None:
        super(MMFF, self).__init__()
        self.args = args
        if args.net_fusion.lower() in ['concat', 'cat', 'concatenate']:
            self.layer = nn.Identity()
        elif args.net_fusion.lower() in ['mlp']:
            self.layer = MLP_fusion(img_feat_channels = 512, blood_feat_channels=64, others_feat_channels=64)
        elif args.net_fusion.lower() in ['attention','att', 'attn']:
            self.layer = AttentionImgFusion(nhead = 4, 
                                            img_feat_channels = 512, blood_feat_channels=64, others_feat_channels=64, 
                                            dim_feedforward = 512, batch_first = True)
        elif args.net_fusion.lower() in ['img','image']:
            self.layer = nn.Identity()
        elif args.net_fusion.lower() in ['blood']:
            self.layer = nn.Identity()
        elif args.net_fusion.lower() in ['others']:
            self.layer = nn.Identity()
            
        # all the things below are just attempts
        elif args.net_fusion.lower() in ['mlp_mixer_c','mlp_mixer_c_only']:
            self.layer = MLP_Mixer_C_only(in_channels=args.net_classifier_inchannel, expansion_factor=2, dropout=0.1)
        elif args.net_fusion.lower() in ['mlp_mixer', 'mixer']:
            token_dims = 1 if args.net_name.lower() in ['resnet',' swin'] else round((args.crop_size[0]/32)**2+1)
            self.layer = MLP_Mixer(token_dims= token_dims, 
                                   channel_dims= args.net_classifier_inchannel, 
                                   token_expansion= 1, channel_expansion=2, dropout = 0.1)
        elif args.net_fusion.lower() in ['attention_mixer', 'att_mixer','attn_mixer']:
            token_dims = 1 if args.net_name.lower() in ['resnet',' swin'] else round((args.crop_size[0]/32)**2+1)
            self.layer = Attention_Mixer(nhead = 4, 
                                          img_feat_channels = 512, blood_feat_channels=64, others_feat_channels=64, 
                                          dim_feedforward = 512, batch_first = True,
                                          token_dims=token_dims,
                                          channel_dims= args.net_classifier_inchannel,
                                          token_expansion=1, channel_expansion=2, dropout = 0.1)
        else:
            raise NotImplementedError('Net fusion module {} is not implemented!'.format(args.net_fusion))
        
    def forward(self, feature_dict):
        feature_list = []
        if 'img_feature'in feature_dict.keys():
            img_feat = feature_dict['img_feature']
            feature_list.append(img_feat)
        if 'blood_feature' in feature_dict.keys():
            blood_feat = feature_dict['blood_feature']
            feature_list.append(blood_feat)
        if 'others_feature' in feature_dict.keys():
            others_feat = feature_dict['others_feature']
            feature_list.append(others_feat)
            
        if self.args.net_fusion.lower() in ['concat', 'cat', 'concatenate']:
            feature = torch.cat(feature_list, dim=2).squeeze(1)
        elif self.args.net_fusion.lower() in ['attention', 'att', 'attn']:
            feature = self.layer(img_feat, blood_feat, others_feat)
        elif self.args.net_fusion.lower() in ['mlp']:
            feature = self.layer(img_feat, blood_feat, others_feat)
        elif self.args.net_fusion.lower() in ['img', 'image']:
            feature = torch.mean(img_feat, dim=1, keepdim=True).squeeze(1)
        elif self.args.net_fusion.lower() in ['blood']:
            feature = blood_feat
        elif self.args.net_fusion.lower() in ['others']:
            feature = others_feat
            
        # attempts
        elif self.args.net_fusion.lower() in ['mlp_mixer_c','mlp_mixer_c_only']:
            feature = self.layer(img_feat, blood_feat, others_feat)
        elif self.args.net_fusion.lower() in ['mlp_mixer', 'mixer']:
            feature = self.layer(img_feat, blood_feat, others_feat)
        elif self.args.net_fusion.lower() in ['attention_mixer', 'att_mixer','attn_mixer']:
            feature = self.layer(img_feat, blood_feat, others_feat)

        return feature

class MHCLS(nn.Module):
    def __init__(self, args, in_channels, mid_channels=512) -> None:
        super(MHCLS, self).__init__()
        self.args = args
        self.num_heads = args.net_nheads
        self.projection = nn.Sequential(nn.Linear(in_channels, mid_channels),
                                        nn.BatchNorm1d(mid_channels),
                                        nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(p=args.net_dropout)
        self.invade_classifiers = nn.ModuleList([nn.Linear(mid_channels, args.net_invade_classes) for i in range(self.num_heads)])
        self.surgery_classifiers = nn.ModuleList([nn.Linear(mid_channels, args.net_surgery_classes) for i in range(self.num_heads)])
        
    def forward(self, x):
        pred_invade = []
        pred_surgery = []
        feat = self.projection(x)
        for i in range(self.num_heads):
            p_invade = self.dropout(feat)
            p_surgery = self.dropout(feat)
            pred_invade.append(self.invade_classifiers[i](p_invade))
            pred_surgery.append(self.surgery_classifiers[i](p_surgery))
        return pred_invade, pred_surgery, p_invade

class Custom_Model_3D(nn.Module):
    def __init__(self,args) -> None:
        super(Custom_Model_3D,self).__init__()
        self.args = args
        # Image Encoder
        if args.net_name.lower() == 'r2plus1d':
            print("Using Image Encoder: r2plus1d")
            if args.net_backbone =='resnet18':
                self.img_encoder = R2Plus1DNet(args, layer_sizes=(2,2,2,2), block_type=SpatioTemporalResBlock) # out channel: 512
            elif args.net_backbone == 'resnet34':
                self.img_encoder = R2Plus1DNet(args, layer_sizes=(3,4,6,3), block_type=SpatioTemporalResBlock) # out channel: 2048
            else:
                raise NotImplementedError("Backbone {} is not implemented!".format(args.net_backbone))
            if args.net_pretrain:
                self.img_encoder.load_state_dict(torch.load(args.net_pretrain), strict=False)
        elif args.net_name.lower() == 'swin':
            print("Using Image Encoder: swin")
            self.img_encoder = SwinExtractor(args)
        elif args.net_name.lower() == 'uniformerv2':
            print("Using Image Encoder: uniformerv2")
            self.img_encoder = UniFormerV2_Extractor(args)
        elif args.net_name.lower() == 'vit':
            print("Using Image Encoder: vit")
            self.img_encoder = ViT_Extractor(args)
        elif args.net_name.lower() in ['none']:
            print("Using Image Encoder: none")
            self.img_encoder = nn.Identity()
        else:
            raise NotImplementedError("Image encoder {} is not implemented!".format(args.net_name.lower()))
        
        # Blood Encoder
        if args.net_blood_name.lower() == 'mlp':
            print("Using Blood Encoder: mlp")
            self.blood_encoder = MLP(args, inchannels=12, outchannels=64, num_layers=2)
        elif args.net_blood_name.lower() in ['transformer']:
            print("Using Blood Encoder: transformer")
            self.blood_encoder = CLIP_Encoder(args.net_text_pretrain, outchannels=64, context_length=256)
        elif args.net_blood_name.lower() in ['none']:
            print("Using Blood Encoder: none")
            self.blood_encoder = nn.Identity()
        else:
            raise NotImplementedError("Blood encoder {} is not implemented!".format(args.net_blood_name.lower()))
        
        # Others Encoder
        if args.net_others_name.lower() == 'mlp':
            print("Using Others Encoder: mlp")
            self.others_encoder = MLP(args, inchannels=7, outchannels=64, num_layers=2)
        elif args.net_others_name.lower() == 'transformer':
            print("Using Others Encoder: transformer")
            self.others_encoder = CLIP_Encoder(args.net_text_pretrain, outchannels=64)
        elif args.net_others_name.lower() in ['none']:
            print("Using Others Encoder: none")
            self.others_encoder = nn.Identity()
        else:
            raise NotImplementedError("Others encoder {} is not implemented!".format(args.net_others_name.lower()))
        
        # Feature Fusion
        self.fusion = MMFF(args)
        
        self.classifier = MHCLS(args, in_channels=args.net_classifier_inchannel)
        self.classifier_ct = MHCLS(args, in_channels=512)
        
        self.img_params = nn.ModuleList([self.img_encoder])
        self.text_params = nn.ModuleList([self.blood_encoder, self.others_encoder])
        self.class_params = nn.ModuleList([self.classifier, self.classifier_ct])
        
    def forward(self, batch):
        feature_dict = {}
        # Encode images
        if not self.args.net_name.lower() in ['none']:
            img_feature, pred_seg, feat_list = self.img_encoder(batch['img'])
            img_feature = img_feature.unsqueeze(1)
            feature_dict['img_feature'] = img_feature
        else:
            pass
        # Encode blood info
        if not self.args.net_blood_name.lower() in ['none']:
            if self.args.net_blood_name.lower() == 'transformer':
                blood_feature = self.blood_encoder(batch['blood_des'])
            else:
                blood_feature = self.blood_encoder(batch['blood'])
            feature_dict['blood_feature'] = blood_feature
        else:
            pass
        # Encode others info
        if not self.args.net_others_name.lower() in ['none']:
            if self.args.net_others_name.lower() == 'transformer':
                others_feature = self.others_encoder(batch['others_des'])
            else:
                others_feature = self.others_encoder(batch['others'])
            feature_dict['others_feature'] = others_feature
        else:
            pass
        
        feature = self.fusion(feature_dict)
        
        pred_invade, pred_surgery, img_feature = self.classifier(feature)
        if not self.args.net_name.lower() in ['none']:
            ct_pred_invade, ct_pred_surgery, ct_img_feature = self.classifier_ct(img_feature)
        else:
            ct_pred_invade, ct_pred_surgery = pred_invade, pred_surgery 
            pred_seg = None
            
        output = {'preds_invade': pred_invade, 
                'preds_surgery': pred_surgery,
                'ct_preds_invade': ct_pred_invade,
                'ct_preds_surgery': ct_pred_surgery,
                'pred_seg': pred_seg,
                'img_feature': img_feature,}
        return output
    
