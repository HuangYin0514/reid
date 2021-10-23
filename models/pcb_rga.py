import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from utils import torchtool
from .rga_module import RGA_Module


class Resnet50_Branch(nn.Module):
    def __init__(self, ** kwargs):
        super(Resnet50_Branch, self).__init__()

        # backbone--------------------------------------------------------------------------
        resnet = models.resnet50(pretrained=True)
        # Modifiy the stride of last conv layer----------------------------
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        # Remove avgpool and fc layer of resnet------------------------------
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

    def forward(self, x):
        return self.backbone(x)


class PCB_RGA(nn.Module):
    def __init__(self, num_classes,  loss='softmax', height=384, width=128, **kwargs):
        super(PCB_RGA, self).__init__()
        self.parts = 6
        self.num_classes = num_classes
        self.loss = loss

        # backbone--------------------------------------------------------------------------
        self.backbone = Resnet50_Branch()

        # rga module--------------------------------------------------------------------------
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
        spa_on = False
        cha_on = True
        s_ratio = 8
        c_ratio = 8
        d_ratio = 8
        self.rga_att = RGA_Module(2048, (height//16)*(width//16), use_spatial=spa_on, use_channel=cha_on,
                                  cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)

        # avgpool--------------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        # self.dropout = nn.Dropout(p=0.5)

        # local_conv--------------------------------------------------------------------
        self.local_conv_list = nn.ModuleList()
        for _ in range(self.parts):
            local_conv = nn.Sequential(
                nn.Conv1d(2048, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True))
            # local_conv.apply(torchtool.weights_init_kaiming)
            self.local_conv_list.append(local_conv)

        # Classifier for each stripe--------------------------------------------------------------------------
        self.fc_list = nn.ModuleList()
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            # fc.apply(torchtool.weights_init_classifier)
            self.fc_list.append(fc)

    def forward(self, x):
        # backbone(Tensor T)------------------------------------------------------------------------------------
        resnet_features = self.backbone(x)

        # rga_att ([N, 2048, 24, 6])------------------------------------------------------------------------------------
        resnet_features = self.rga_att(resnet_features)

        # tensor g([N, 2048, 6, 1])---------------------------------------------------------------------------------
        features_G = self.avgpool(resnet_features)

        # 1x1 conv([N, C=256, H=6, W=1])---------------------------------------------------------------------------------
        features_H = []
        for i in range(self.parts):
            stripe_features_H = self.local_conv_list[i](features_G[:, :, i, :])
            features_H.append(stripe_features_H)

        # Return the features_H([N,1536])***********************************************************************
        if not self.training:
            v_g = torch.cat(features_H, dim=2)
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)

        # fc（[N, C=num_classes]）---------------------------------------------------------------------------------
        batch_size = x.size(0)
        logits_list = [self.fc_list[i](features_H[i].view(batch_size, -1)) for i in range(self.parts)]

        return logits_list


def pcb_rga(num_classes, **kwargs):
    return PCB_RGA(
        num_classes=num_classes,
        **kwargs
    )
