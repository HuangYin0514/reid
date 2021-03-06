
import torch
import torch.nn as nn
from .backbones.resnet50 import resnet50
import torch.nn.functional as F
from .model_utils.init_param import weights_init_classifier, weights_init_kaiming
from .layers.rga_module import RGA_Module


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
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        return x


class Feature_Fusion_Module(nn.Module):
    # 自定义特征融合模块
    def __init__(self, parts, **kwargs):
        super(Feature_Fusion_Module, self).__init__()

        self.parts = parts

        self.fc1 = nn.Linear(256, 6)
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

class pcb_ffm(nn.Module):
    def __init__(self, num_classes, loss='softmax', ** kwargs):

        super(pcb_ffm, self).__init__()
        self.parts = 6
        self.num_classes = num_classes
        self.loss = loss
        ########################################################################################################
        # backbone--------------------------------------------------------------------------
        self.backbone = Resnet_Backbone()

        ########################################################################################################
        # feature fusion module--------------------------------------------------------------------------
        self.ffm = Feature_Fusion_Module(self.parts)

        ########################################################################################################
        # gloab--------------------------------------------------------------------------
        self.k11_conv = nn.Conv2d(2048, 512, kernel_size=1)
        self.gloab_agp = nn.AdaptiveAvgPool2d((1, 1))
        self.gloab_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
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
        resnet_features = self.backbone(x)  # ([N, 2048, 24, 8])

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