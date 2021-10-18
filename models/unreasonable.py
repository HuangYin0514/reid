import torch
import torch.nn as nn
from .backbones.resnet50 import resnet50
import torch.nn.functional as F


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

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        # x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        return x


class pcb(nn.Module):
    def __init__(self, num_classes):

        self.parts = 6

        super(pcb, self).__init__()

        # backbone
        self.backbone = Resnet_Backbone()

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size = x.size(0)

        # backbone(Tensor T)
        resnet_features = self.backbone(x)
        global_feat = self.gap(resnet_features)
        global_feat = global_feat.view(batch_size, -1)

        if self.training:

            return resnet_features, global_feat
        else:

            return resnet_features, global_feat
