import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.resnet50 import resnet50
from .model_utils.init_param import weights_init_classifier, weights_init_kaiming


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


class DropBlock2D(nn.Module):
    def __init__(self, keep_prob=0.9, block_size=7, beta=0.9):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.beta = beta

    def normalize(self, input):
        min_c, max_c = input.min(1, keepdim=True)[0], input.max(1, keepdim=True)[0]
        input_norm = (input - min_c) / (max_c - min_c + 1e-8)
        return input_norm

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1.0 - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.conv2d(
            M,
            torch.ones((input.shape[1], 1, self.block_size, self.block_size)).to(
                device=input.device, dtype=input.dtype
            ),
            padding=self.block_size // 2,
            groups=input.shape[1],
        )

        Msum = (Msum < 1).to(device=input.device, dtype=input.dtype)
        input2 = input * Msum
        x_norm = self.normalize(input2)
        mask = (x_norm > self.beta).float()
        block_mask = 1 - (mask * x_norm)
        return input * block_mask


class MFRL(nn.Module):
    def __init__(self):
        super(MFRL, self).__init__()

        self.nums = 3
        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(
                nn.Conv2d(
                    448,
                    448,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            bns.append(nn.BatchNorm2d(448))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        spx = torch.split(x, 448, dim=1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        ms = torch.split(out, 448, dim=1)
        m12 = torch.cat([ms[0], ms[1]], dim=1)
        m34 = torch.cat([ms[2], ms[3]], dim=1)

        return m12, m34


class pcb_module(nn.Module):
    def __init__(self, num_classes, in_channel):
        super(pcb_module, self).__init__()

        self.num_classes = num_classes
        self.parts = 6

        self.avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))

        self.local_conv_list = nn.ModuleList()
        for _ in range(self.parts):
            local_conv = nn.Sequential(
                nn.Conv1d(in_channel, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )
            self.local_conv_list.append(local_conv)

        # Classifier for each stripe （parts feature）
        self.parts_classifier_list = nn.ModuleList()
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.parts_classifier_list.append(fc)

    def forward(self, x):
        batch_size = x.size(0)

        features_G = self.avgpool(x)

        features_H = []  # contains 6 ([N, 256, 1])
        for i in range(self.parts):
            stripe_features_H = self.local_conv_list[i](features_G[:, :, i, :])
            features_H.append(stripe_features_H)

        if self.training:
            parts_score_list = [
                self.parts_classifier_list[i](features_H[i].view(batch_size, -1))
                for i in range(self.parts)
            ]  # shape list（[N, C=num_classes]）

            return features_H, parts_score_list
        else:
            return features_H


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
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

        # apnet模块
        self.level = 2
        self.att1 = SELayer(64, 8)
        self.att2 = SELayer(256, 32)
        self.att3 = SELayer(512, 64)
        self.att4 = SELayer(1024, 128)
        self.att5 = SELayer(2048, 256)

        self.att_s1 = SAMS(64, int(64 / self.level), radix=self.level)
        self.att_s2 = SAMS(256, int(256 / self.level), radix=self.level)
        self.att_s3 = SAMS(512, int(512 / self.level), radix=self.level)
        self.att_s4 = SAMS(1024, int(1024 / self.level), radix=self.level)
        self.att_s5 = SAMS(2048, int(2048 / self.level), radix=self.level)

        self.BN1 = BN2d(64)
        self.BN2 = BN2d(256)
        self.BN3 = BN2d(512)
        self.BN4 = BN2d(1024)
        self.BN5 = BN2d(2048)

        self.att_ss1 = SAMS(64, int(64 / self.level), radix=self.level)
        self.att_ss2 = SAMS(256, int(256 / self.level), radix=self.level)
        self.att_ss3 = SAMS(512, int(512 / self.level), radix=self.level)
        self.att_ss4 = SAMS(1024, int(1024 / self.level), radix=self.level)
        self.att_ss5 = SAMS(2048, int(2048 / self.level), radix=self.level)

        self.BN_1 = BN2d(64)
        self.BN_2 = BN2d(256)
        self.BN_3 = BN2d(512)
        self.BN_4 = BN2d(1024)
        self.BN_5 = BN2d(2048)

        self.db1 = DropBlock2D(keep_prob=0.9, block_size=7)
        self.db2 = DropBlock2D(keep_prob=0.9 ,block_size=7)
        self.db3 = DropBlock2D(keep_prob=0.9 ,block_size=7)

        self.avgpool1 = nn.AdaptiveAvgPool2d((6, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((6, 1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((6, 1))
        self.avgpool4 = nn.AdaptiveAvgPool2d((6, 1))

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        x = self.resnet_layer1(x)
        x = self.att_ss2(x)
        x = self.BN_2(x)
        x = self.att_s2(x)
        x = self.BN2(x)
        y = self.att2(x)
        x = x * y.expand_as(x)

        y1 = self.db1(x)
        avg_y1 = self.avgpool1(y1)
        # avg_y1 = self.avgpool1(x)

        x = self.resnet_layer2(x)
        x = self.att_ss3(x)
        x = self.BN_3(x)
        x = self.att_s3(x)
        x = self.BN3(x)
        y = self.att3(x)
        x = x * y.expand_as(x)

        y2=self.db2(x)
        avg_y2 = self.avgpool2(y2)

        x = self.resnet_layer3(x)
        x = self.att_ss4(x)
        x = self.BN_4(x)
        x = self.att_s4(x)
        x = self.BN4(x)
        y = self.att4(x)
        x = x * y.expand_as(x)

        layer3_f = x

        y3 = self.db3(x)
        avg_y3 = self.avgpool3(y3)

        x = self.resnet_layer4(x)
        x = self.att_ss5(x)
        x = self.BN_5(x)
        x = self.att_s5(x)
        x = self.BN5(x)
        y = self.att5(x)
        x = x * y.expand_as(x)

        avg_out = torch.cat([avg_y1, avg_y2, avg_y3], dim=1)

        return x, avg_out, layer3_f


class baseline_apne_drop(nn.Module):
    def __init__(self, num_classes):

        self.num_classes = num_classes

        super(baseline_apne_drop, self).__init__()

        # backbone
        self.backbone = Resnet_Backbone()

        # baseline
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        # part(pcb）--------------------------------------------------------------------------

        self.mfm = MFRL()

        self.pcb1 = pcb_module(num_classes, in_channel=896)
        self.pcb2 = pcb_module(num_classes, in_channel=896)
        self.pcb3 = pcb_module(num_classes, in_channel=1024)

    def forward(self, x):
        batch_size = x.size(0)

        resnet_features, avg_out,layer3_f = self.backbone(x)  # (batch_size, 2048, 16, 8)

        # baseline
        x = self.avgpool(resnet_features)  # (batch_size, 2048, 1, 1)
        x = x.view(batch_size, -1)  # (batch_size, 2048)
        feat = self.bottleneck(x)  # (batch_size, 2048)

        hight_f, low_f = self.mfm(avg_out)

        

        if self.training:
            score = self.classifier(feat)  # (batch_size, num_classes)

            # parts --------------------------------------------------------------------------
            features_H1, parts_score_list1 = self.pcb1(hight_f)
            features_H2, parts_score_list2 = self.pcb2(low_f)
            features_H3, parts_score_list3 = self.pcb3(layer3_f)

            return score, x, parts_score_list1, parts_score_list2,parts_score_list3
        else:
            return feat
