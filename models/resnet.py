
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class ResnetModel(nn.Module):
    def __init__(self,
                 num_classes,
                 loss='softmax',
                 ** kwargs):

        super(ResnetModel, self).__init__()
        self.num_classes = num_classes
        self.loss = loss

        resnet = models.resnet50(pretrained=True)
        # Modifiy the stride of last conv layer
        resnet.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=3, bias=False, stride=1, padding=1)
        resnet.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048))

        # Remove avgpool and fc layer of resnet
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # Add new layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classfier = nn.Linear(2048, num_classes)

    def forward(self, x):
        resnet_features = self.backbone(x)
        features = self.avgpool(resnet_features)
        features = features.view(x.size(0), -1)
        logits = self.classfier(features)

        if not self.training:
            return features

        if self.loss == 'softmax':
            return logits
        elif self.loss == 'triplet':
            return logits, features.view(x.size(0), -1)
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        return logits_list




##########
# Instantiation
##########

        
def Res_net(num_classes,  **kwargs):
    return ResnetModel(
        num_classes=num_classes,
        **kwargs
    )
