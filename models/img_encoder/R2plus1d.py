import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch.nn.functional as F


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, first_conv=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        if first_conv:
            # decomposing the parameters into spatial and temporal components by
            # masking out the values with the defaults on the axis that
            # won't be convolved over. This is necessary to avoid unintentional
            # behavior such as padding being added twice
            spatial_kernel_size = kernel_size
            spatial_stride = (1, stride[1], stride[2])
            spatial_padding = padding

            temporal_kernel_size = (3, 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (1, 0, 0)

            # from the official code, first conv's intermed_channels = 45
            intermed_channels = 45

            # the spatial conv is effectively a 2D conv due to the
            # spatial_kernel_size, followed by batch_norm and ReLU
            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                          stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)
            # the temporal conv is effectively a 1D conv, but has batch norm
            # and ReLU added inside the model constructor, not here. This is an
            # intentional design choice, to allow this module to externally act
            # identical to a standard Conv3D, so it can be reused easily in any
            # other codebase
            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                           stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()
        else:
            # decomposing the parameters into spatial and temporal components by
            # masking out the values with the defaults on the axis that
            # won't be convolved over. This is necessary to avoid unintentional
            # behavior such as padding being added twice
            spatial_kernel_size =  (1, kernel_size[1], kernel_size[2])
            spatial_stride =  (1, stride[1], stride[2])
            spatial_padding =  (0, padding[1], padding[2])

            temporal_kernel_size = (kernel_size[0], 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (padding[0], 0, 0)

            # compute the number of intermediary channels (M) using formula
            # from the paper section 3.5
            intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                                (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

            # the spatial conv is effectively a 2D conv due to the
            # spatial_kernel_size, followed by batch_norm and ReLU
            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                        stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)

            # the temporal conv is effectively a 1D conv, but has batch norm
            # and ReLU added inside the model constructor, not here. This is an
            # intentional design choice, to allow this module to externally act
            # identical to a standard Conv3D, so it can be reused easily in any
            # other codebase
            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                        stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()



    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.relu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R2Plus1D_Decoder(nn.Module):
    def __init__(self, args, in_channels) -> None:
        super(R2Plus1D_Decoder,self).__init__()
        self.upconv1= nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=int(in_channels/2), kernel_size=3, padding=1),
            nn.BatchNorm3d(int(in_channels/2)),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=int(in_channels/4), kernel_size=3, padding=1),
            nn.BatchNorm3d(int(in_channels/4)),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.Conv3d(in_channels=int(in_channels/2), out_channels=int(in_channels/8), kernel_size=3, padding=1),
            nn.BatchNorm3d(int(in_channels/8)),
            nn.ReLU(inplace=True)
        )
        self.upconv4 = nn.Sequential(
            nn.Conv3d(in_channels=int(in_channels/4), out_channels=int(in_channels/16), kernel_size=3, padding=1),
            nn.BatchNorm3d(int(in_channels/16)),
            nn.ReLU(inplace=True)
        )
        self.seg_head = nn.Sequential(
            nn.Conv3d(in_channels=int(in_channels/16), out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, args.net_seg_classes, kernel_size=1)
        )
        
    def forward(self, feat_list):
        feat1,feat2,feat3,feat4 = feat_list
        output_feat_list = []
        feat1 = self.upconv1(feat1) # C/2 x T/16 x H/16 x W/16
        output_feat_list.append(feat1)
        feat1 = F.interpolate(feat1, scale_factor=2, mode='trilinear') # C/2 x T/8 x H/8 x W/8
        feat2 = torch.cat((feat1, feat2), dim=1) # C x T/8 x H/8 x W/8
        feat2 = self.upconv2(feat2) # C/4 x T/8 x H/8 x W/8
        output_feat_list.append(feat2)
        feat2 = F.interpolate(feat2, scale_factor=2, mode='trilinear') # C/4 x T/4 x H/4 x W/4
        feat3 = torch.cat((feat2,feat3), dim=1) # C/2 x T/4 x H/4 x W/4
        feat3 = self.upconv3(feat3) # C/8 x T/4 x H/4 x W/4
        output_feat_list.append(feat3)
        feat3 = F.interpolate(feat3, scale_factor=2, mode='trilinear') # C/8 x T/2 x H/2 x W/2
        feat4 = torch.cat((feat3,feat4), dim=1) # C/4 x T/2 x H/2 x W/2
        feat4 = self.upconv4(feat4) # C/16 x T/2 x H/2 x W/2
        output_feat_list.append(feat4)
        feat = self.seg_head(feat4)
        pred_seg = F.interpolate(feat, scale_factor=2, mode='trilinear')
        
        return pred_seg, output_feat_list
        

class R2Plus1DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, args, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()
        self.args=args

        # first conv, with stride 1x2x2 and kernel size 1x7x7
        self.conv1 = SpatioTemporalConv(1, 64, (3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3), first_conv=True)
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveMaxPool3d(1)
        if args.net_seg_celoss or args.net_seg_diceloss:
            self.decoder = R2Plus1D_Decoder(args, in_channels=512)
            
        # self.__init_weight__()
            
    def __init_weight__(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        feat4 = self.conv2(x)
        feat3 = self.conv3(feat4)
        feat2 = self.conv4(feat3)
        feat1 = self.conv5(feat2)
        if self.args.net_seg_celoss or self.args.net_seg_diceloss:
            pred_seg, feat_list = self.decoder([feat1,feat2,feat3,feat4])
        else:
            pred_seg = None
            feat_list = [feat1, feat2, feat3, feat4]
        x = self.pool(feat1)

        return x.view(-1, 512), pred_seg, feat_list


class MHCLS(nn.Module):
    def __init__(self, args) -> None:
        super(MHCLS, self).__init__()
        self.args = args
        self.num_heads = args.net_nheads
        self.invade_classifiers = nn.ModuleList([nn.Linear(512, args.net_invade_classes) for i in range(self.num_heads)])
        self.surgery_classifiers = nn.ModuleList([nn.Linear(512, args.net_surgery_classes) for i in range(self.num_heads)])
        
    def forward(self, x):
        pred_invade = []
        pred_surgery = []
        for i in range(self.num_heads):
            pred_invade.append(self.invade_classifiers[i](x))
            pred_surgery.append(self.surgery_classifiers[i](x))
        return pred_invade, pred_surgery


class R2Plus1DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, args):
        super(R2Plus1DClassifier, self).__init__()

        if args.net_backbone =='resnet18':
            self.res2plus1d = R2Plus1DNet(layer_sizes=(2,2,2,2), block_type=SpatioTemporalResBlock)
        elif args.net_backbone == 'resnet50':
            self.res2plus1d = R2Plus1DNet(layer_sizes=(3,4,6,3), block_type=SpatioTemporalResBlock)
        else:
            raise NotImplementedError("Backbone {} is not Implemented!".format(args.net_backbone))
        self.classifier = MHCLS(args)

        self.__init_weight()
        self.pretrained = nn.ModuleList([self.res2plus1d])
        self.new_added = nn.ModuleList([self.classifier])

    def forward(self, x):
        x = self.res2plus1d(x)
        pred_invade, pred_surgery = self.classifier(x)

        return pred_invade, pred_surgery

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res2plus1d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.MHCLS]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k
