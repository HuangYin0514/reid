import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.resnet50 import resnet50
from .model_utils.init_param import weights_init_classifier, weights_init_kaiming
from .layers.pc import PC_Module
from .layers.nn_utils import *


class SAMS(nn.Module):
    """
    Split-Attend-Merge-Stack agent
    Input an feature map with shape H*W*C, we first split the feature maps into
    multiple parts, obtain the attention map of each part, and the attention map
    for the current pyramid level is constructed by mergiing each attention map.
    """

    def __init__(
        self,
        in_channels,
        channels,
        radix=4,
        reduction_factor=4,
        norm_layer=nn.BatchNorm2d,
    ):
        super(SAMS, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=1)

    def forward(self, x):

        batch, channel = x.shape[:2]
        splited = torch.split(x, channel // self.radix, dim=1)

        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        atten = torch.split(atten, channel // self.radix, dim=1)

        out = torch.cat([att * split for (att, split) in zip(atten, splited)], 1)
        return out.contiguous()


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class BN2d(nn.Module):
    def __init__(self, planes):
        super(BN2d, self).__init__()
        self.bottleneck2 = nn.BatchNorm2d(planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck2(x)


# apnet模块
class apnet(nn.Module):
    def __init__(self, in_channels, out_planes):
        super(apnet, self).__init__()

        self.level = 2
        self.att = SELayer(in_channels, out_planes)
        self.att_s = SAMS(in_channels, int(in_channels / self.level), radix=self.level)
        self.BN = BN2d(in_channels)
        self.att_ss = SAMS(in_channels, int(in_channels / self.level), radix=self.level)
        self.BN_ = BN2d(in_channels)

    def forward(self, x):
        x = self.att_ss(x)
        x = self.BN_(x)
        x = self.att_s(x)
        x = self.BN(x)
        y = self.att(x)
        x = x * y.expand_as(x)
        return x


# FPB
class FPNModule(nn.Module):
    def __init__(self, num_layers, num_channels):
        super(FPNModule, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.eps = 0.0001

        self.convs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for _ in range(4):
            conv = Conv_Bn_Relu(self.num_channels, self.num_channels, k=3, p=1)
            self.convs.append(conv)

        for _ in range(2):
            downsample = nn.Sequential(
                nn.Conv2d(self.num_channels, self.num_channels, 1, bias=False),
                nn.BatchNorm2d(self.num_channels),
                nn.ReLU(inplace=True),
            )
            self.downsamples.append(downsample)

        self.pc1 = PC_Module(self.num_channels, dropout=True)

        self._init_params()

    def _init_params(self):
        for downsample in self.downsamples:
            init_struct(downsample)

        return

    def forward(self, x):

        reg_feats = []

        y = x
        x_clone = []
        for t in x:
            x_clone.append(t.clone())

        reg_feats.append(y[0])
        y[0] = self.convs[0](y[0])
        reg_feat = self.pc1(y[1])
        reg_feats.append(reg_feat)
        y[1] = self.convs[1](
            reg_feat + F.interpolate(y[0], scale_factor=2, mode="nearest")
        )
        y[1] = self.convs[2](y[1]) + self.downsamples[0](x_clone[1])
        y[0] = self.convs[3](
            y[0] + F.max_pool2d(y[1], kernel_size=2)
        ) + self.downsamples[1](x_clone[0])

        return y, reg_feats


class FPN(nn.Module):
    def __init__(self, num_layers, in_channels):
        super(FPN, self).__init__()

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.num_neck_channel = 256

        self.lateral_convs = nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = Conv_Bn_Relu(in_channels[i], self.num_neck_channel, k=1)
            self.lateral_convs.append(conv)

        self.fpn_module1 = FPNModule(self.num_layers, self.num_neck_channel)

        self.conv = Conv_Bn_Relu(
            self.num_neck_channel, self.in_channels[1], k=1, activation_cfg=False
        )
        self.relu = nn.ReLU(inplace=True)

        self._init_params()

    def _init_params(self):

        return

    def forward(self, x):
        y = []
        for i in range(self.num_layers):
            y.append(self.lateral_convs[i](x[i + 1]))

        y, reg_feat = self.fpn_module1(y)

        y = self.conv(y[0])  #
        y = self.relu(y)

        return y, reg_feat


# apnet修改的模块
class Resnet_Backbone(nn.Module):
    def __init__(self):
        super(Resnet_Backbone, self).__init__()

        # backbone--------------------------------------------------------------------------
        # change the model different from pcb
        resnet = resnet50(pretrained=True)
        # Modifiy the stride of last conv layer----------------------------
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        # Remove avgpool and fc layer of resnet------------------------------
        self.resnet_conv1 = resnet.conv1
        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool

        self.resnet_layer0 = nn.Sequential(
            self.resnet_conv1, self.resnet_bn1, self.resnet_relu, self.resnet_maxpool
        )
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

        # apent
        self.apnet1 = apnet(256, 32)
        self.apnet2 = apnet(512, 64)
        self.apnet3 = apnet(1024, 128)
        self.apnet4 = apnet(2048, 256)

        # FPB
        self.pc1 = PC_Module(512, dropout=True)

    def forward(self, x):

        y_l0 = self.resnet_layer0(x)

        y_l1 = self.resnet_layer1(y_l0)
        y_l1 = self.apnet1(y_l1)

        y_l2 = self.resnet_layer2(y_l1)
        y_l2 = self.apnet2(y_l2)

        y_l2_1 = self.pc1(y_l2)

        y_l3 = self.resnet_layer3(y_l2)
        y_l3 = self.apnet3(y_l3)

        y_l4 = self.resnet_layer4(y_l3)
        y_l4 = self.apnet4(y_l4)

        fs = []
        fs.append(y_l4)
        fs.append(y_l3)
        fs.append(y_l2)
        fs.append(y_l1)

        return fs, y_l2_1


class baseline_apnet_FPB(nn.Module):
    def __init__(self, num_classes):

        self.num_classes = num_classes
        self.branch_layers = 2
        self.num_parts = 3

        super(baseline_apnet_FPB, self).__init__()

        # backbone
        self.backbone = Resnet_Backbone()

        # baseline
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        # fpb
        self.in_channels = [2048, 1024, 512, 256]
        self.neck = FPN(self.branch_layers, self.in_channels)

        self.part_pools = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        self.dim_reds = DimReduceLayer(self.in_channels[1], 256)
        self.classifiers = nn.ModuleList(
            [nn.Linear(256, num_classes) for _ in range(self.num_parts)]
        )

    def cross_ofp(self, x):
        x[1] = F.max_pool2d(x[1], kernel_size=2)
        y = torch.cat(x, 1)
        return y

    def forward(self, x):
        batch_size = x.size(0)

        fs, f_l2_1 = self.backbone(x)  # (batch_size, 2048, 16, 8)
        f_l4, f_l3, f_l2, f_l1 = fs

        # baseline
        f_l4 = self.avgpool(f_l4)  # (batch_size, 2048, 1, 1)
        f_l4_train = f_l4.view(batch_size, -1)  # (batch_size, 2048)
        f_l4 = self.bottleneck(f_l4_train)  # (batch_size, 2048)

        # fpb
        # f_branch(bs, 2048,16,8)  # reg_feats [(batch_size, 2048,16,8),(batch_size, 2048,32,16)]
        f_branch, reg_feats = self.neck(fs)
        f_parts = self.part_pools(f_branch)  # (batch_size,1024,3,1)

        if self.training:
            y = [] #全局+局部损伤
            score = self.classifier(f_l4)  # (batch_size, num_classes)
            y.append(score)

            f_short = self.dim_reds(f_parts)
            for j in range(self.num_parts):
                f_j = f_short[:, :, j, :].view(batch_size, -1)
                y_j = self.classifiers[j](f_j)
                y.append(y_j)

            
            reg_feat_re = [] #正则化损失
            reg_feat_re.append(self.cross_ofp(reg_feats))

            f = [] # 三元组损失
            f.append(F.normalize(f_l4_train, p=2, dim=1).view(batch_size, -1))
            f.append(F.normalize(f_parts, p=2, dim=1).view(batch_size, -1))

            f = torch.cat(f, 1)

            return y, f, reg_feat_re
        else:

            f = []
            f.append(F.normalize(f_l4, p=2, dim=1))
            f.append(F.normalize(f_parts, p=2, dim=1).view(batch_size, -1))

            f = torch.cat(f, dim=1)

            return f
