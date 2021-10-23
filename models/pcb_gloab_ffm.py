#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :pcb_gloab_rga.py
@说明        :增加rga模块，测试
@时间        :2020/10/11 17:03:49
@作者        :HuangYin
@版本        :1.0
'''

import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
from .rga_module import RGA_Module


__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


class Resnet50_backbone(nn.Module):
    def __init__(self, ** kwargs):
        super(Resnet50_backbone, self).__init__()

        # backbone--------------------------------------------------------------------------
        # change the model different from pcb
        resnet = resnet50_cbam(pretrained=True)
        # Modifiy the stride of last conv layer----------------------------
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        # Remove avgpool and fc layer of resnet------------------------------
        self.resnet_conv1 = resnet.conv1
        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        layer_2_out = x
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)
        return x, layer_2_out


def custom_RGA_Module():
    # 自定义 RGA 模块
    branch_name = 'rgas'
    if 'rgasc' in branch_name:
        spa_on = True
        cha_on = True
    elif 'rgas' in branch_name:
        spa_on = True
        cha_on = False
    elif 'rgac' in branch_name:
        spa_on = False
        cha_on = True
    else:
        raise NameError

    s_ratio = 8
    c_ratio = 8
    d_ratio = 8

    return RGA_Module(512, 24*8, use_spatial=spa_on, use_channel=cha_on,
                      cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)


class Feature_Fusion_Module(nn.Module):
    # 自定义特征融合模块
    def __init__(self, parts, **kwargs):
        super(Feature_Fusion_Module, self).__init__()

        self.parts = parts

        self.fc1 = nn.Linear(512, 6)
        self.fc1.apply(weights_init_kaiming)

    def forward(self, gloab_feature, parts_features):
        batch_size = gloab_feature.size(0)

        ########################################################################################################
        # compute the weigth of parts features --------------------------------------------------
        w_of_parts = torch.sigmoid(self.fc1(gloab_feature))

        ########################################################################################################
        # compute the features,with weigth --------------------------------------------------
        weighted_feature = torch.zeros_like(parts_features[0])
        for i in range(self.parts):
            new_feature = parts_features[i] * w_of_parts[:, i].view(batch_size, 1, 1).expand(parts_features[i].shape)
            weighted_feature += new_feature

        return weighted_feature.squeeze()


class resnet50_reid(nn.Module):
    def __init__(self, num_classes, loss='softmax', ** kwargs):

        super(resnet50_reid, self).__init__()
        self.parts = 6
        self.num_classes = num_classes
        self.loss = loss
        ########################################################################################################
        # backbone--------------------------------------------------------------------------
        self.backbone = Resnet50_backbone()

        ########################################################################################################
        # feature fusion module--------------------------------------------------------------------------
        self.ffm = Feature_Fusion_Module(self.parts)

        ########################################################################################################
        # gloab--------------------------------------------------------------------------
        self.k11_conv = nn.Conv2d(2048, 512, kernel_size=1)
        self.gloab_agp = nn.AdaptiveAvgPool2d((1, 1))
        self.gloab_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.gloab_conv.apply(weights_init_kaiming)

        self.rga_att = custom_RGA_Module()

        ########################################################################################################
        # part(pcb）--------------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.local_conv_list = nn.ModuleList()
        for _ in range(self.parts):
            local_conv = nn.Sequential(
                nn.Conv1d(2048, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True))
            self.local_conv_list.append(local_conv)

        # Classifier for each stripe （parts feature）-------------------------------------
        self.parts_classifier_list = nn.ModuleList()
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.parts_classifier_list.append(fc)

        ########################################################################################################
        # fusion_feature_classifier（fusion feature）--------------------------------------------------------------------------
        self.fusion_feature_classifier = nn.Linear(256, num_classes)
        nn.init.normal_(self.fusion_feature_classifier.weight, std=0.001)
        nn.init.constant_(self.fusion_feature_classifier.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        ######################################################################################################################
        # backbone(Tensor T) --------------------------------------------------------------------------
        resnet_features, _ = self.backbone(x)  # ([N, 2048, 24, 8])

        # gloab([N, 512]) --------------------------------------------------------------------------
        gloab_features = self.k11_conv(resnet_features)
        gloab_features = self.rga_att(gloab_features)
        gloab_features = self.gloab_agp(gloab_features).view(batch_size, 512, -1)  # ([N, 512, 1])
        gloab_features = self.gloab_conv(gloab_features).squeeze()  # ([N, 512])

        # parts --------------------------------------------------------------------------
        features_G = self.avgpool(resnet_features)  # tensor g([N, 2048, 6, 1])
        features_H = []  # contains 6 ([N, 256, 1])
        for i in range(self.parts):
            stripe_features_H = self.local_conv_list[i](features_G[:, :, i, :])
            features_H.append(stripe_features_H)

        # feature fusion module--------------------------------------------------------------------------
        fusion_feature = self.ffm(gloab_features, features_H)

        ######################################################################################################################
        # Return the features_H if inference--------------------------------------------------------------------------
        if not self.training:
            # features_H.append(gloab_features.unsqueeze_(2))  # ([N,1536+512])
            v_g = torch.cat(features_H, dim=1)
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)

        ######################################################################################################################
        # classifier(parts)--------------------------------------------------------------------------
        parts_score_list = [self.parts_classifier_list[i](features_H[i].view(batch_size, -1)) for i in range(self.parts)]  # shape list（[N, C=num_classes]）

        # classifier(fusion feature)--------------------------------------------------------------------------
        # fusion_score = self.fusion_feature_classifier(fusion_feature)

        return parts_score_list, gloab_features, fusion_feature


# resnet50_cbam_reid_model(return function)-->resnet50_cbam_reid-->Resnet50_backbone(reid backbone)
#       -->resnet50_cbam(return function)-->ResNet-->Bottleneck(or chose the BasicBlock)-->ChannelAttention-->SpatialAttention
def pcb_gloab_ffm(num_classes, **kwargs):
    return resnet50_reid(
        num_classes=num_classes,
        **kwargs
    )
