
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.inplanes = inplanes
        self.planes = planes
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, strides=(2, 2, 2, 2), dilations=(1, 1, 1, 1)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])

        self.inplanes = 1024

        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, 1000)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, dilation=1)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_dict=model_zoo.load_url(model_urls['resnet18'])# Modify 'model_dir' according to your own path
        model.load_state_dict(pretrained_dict, strict=False)
        print('Petrain Model Have been loaded!')
    return model

def resnet34(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict=model_zoo.load_url(model_urls['resnet34'])# Modify 'model_dir' according to your own path
        model.load_state_dict(pretrained_dict, strict=False)
        print('Petrain Model Have been loaded!')
    return model

def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        model.load_state_dict(state_dict, strict=False)
        print("model pretrained initialized")
    return model

class ResNet_Decoder(nn.Module):
    def __init__(self, args, in_channels) -> None:
        super(ResNet_Decoder,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels/2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(in_channels/2)),
            nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels/4), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(in_channels/4)),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels/2), out_channels=int(in_channels/8), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(in_channels/8)),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels/4), out_channels=int(in_channels/16), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(in_channels/16)),
            nn.ReLU(inplace=True)
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels/16), out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=args.net_seg_classes, kernel_size=1)
        )
        
    def forward(self, feat_list):
        feat4, feat3 ,feat2, feat1 = feat_list
        feat1 = self.conv_layer(feat1)
        feat2 = torch.cat((feat1, feat2), dim=1)    # 2C x H/32 x W/32
        feat2 = self.upconv1(feat2) # C/2 x H/32 x W/32
        feat2 = F.interpolate(feat2, scale_factor=2, mode='bilinear') # C/2 x H/16 x W/16
        feat3 = torch.cat((feat2, feat3), dim=1) # C x H/16 x W/16
        feat3 = self.upconv2(feat3) # C/4 x H/16 x W/16
        feat3 = F.interpolate(feat3, scale_factor=2, mode='bilinear') # C/4 x H/8 x W/8
        feat4 = torch.cat((feat3, feat4), dim=1) # C/2 x H/8 x W/8
        feat4 = self.upconv3(feat4) # C/8 x H/8 x W/8
        feat4 = F.interpolate(feat4, scale_factor=2, mode='bilinear') # C/8 x H/4 x W/4
        feat = self.seg_head(feat4)
        pred_seg = F.interpolate(feat, scale_factor=2, mode='bilinear')
        return pred_seg

class ResNet_Extractor(nn.Module):
    def __init__(self, args) -> None:
        super(ResNet_Extractor,self).__init__()
        self.args = args
        if args.net_backbone.lower() in ['resnet18', 'res18']:
            self.resnet = resnet18(pretrained=True, strides=(2,2,2,1), dilations=(1,1,1,2))
            self.layer = nn.Sequential(nn.Linear(512, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True))
            self.decoder = ResNet_Decoder(args, in_channels=512)
        elif args.net_backbone.lower() in ['resnet34', 'res34']:
            self.resnet = resnet34(pretrained=True, strides=(2,2,2,1), dilations=(1,1,1,2))
            self.layer = nn.Sequential(nn.Linear(512, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True))
            self.decoder = ResNet_Decoder(args, in_channels=512)
        elif args.net_backbone.lower() in ['resnet50', 'res50']:
            self.resnet = resnet50(pretrained=True, strides=(2,2,2,1), dilations=(1,1,1,2))
            self.layer = nn.Sequential(nn.Linear(2048, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True))
            self.decoder = ResNet_Decoder(args, in_channels=2048)
            
        self.stem = nn.Sequential(self.resnet.conv1, self.resnet.bn1,self.resnet.relu, self.resnet.maxpool)
        self.stage1 = nn.Sequential(self.resnet.layer1)
        self.stage2 = nn.Sequential(self.resnet.layer2)
        self.stage3 = nn.Sequential(self.resnet.layer3)
        self.stage4 = nn.Sequential(self.resnet.layer4)
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        
        self.pretrained = nn.ModuleList([self.stem, self.stage1, self.stage2, self.stage3, self.stage4])
        self.new_added = nn.ModuleList([self.layer, self.decoder])
        
        if args.net_pretrain:
            state_dict = self.state_dict()
            pretrained_state_dict = torch.load(args.net_pretrain)
            new_state_dict = {}
            for k,v in pretrained_state_dict.items():
                new_key = k.replace("img_encoder.", "")
                new_state_dict[new_key] = v
            self.load_state_dict(new_state_dict, strict=True)
        
    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        x5 = self.stage4(x4)

        feat = self.maxpool(x5).squeeze(2).squeeze(2)
        feat = self.layer(feat).unsqueeze(1)
        
        pred_seg = self.decoder([x2,x3,x4,x5])
        
        return feat, pred_seg
