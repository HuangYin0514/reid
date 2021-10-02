import torch
import torch.nn as nn
from .model_utils.resnet50 import resnet50
from .model_utils.init_param import weights_init_classifier, weights_init_kaiming
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
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        return x


class Feature_Fusion_Module(nn.Module):
    # 自定义特征融合模块
    def __init__(self, parts):
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
            new_feature = parts_features[i] * w_of_parts[:, i].view(
                batch_size, 1, 1
            ).expand(parts_features[i].shape)
            weighted_feature += new_feature

        return weighted_feature.squeeze()


class pcb_bilstm_3branch(nn.Module):
    def __init__(self, num_classes):
        super(pcb_bilstm_3branch, self).__init__()

        self.parts = 6

        # backbone
        self.backbone = Resnet_Backbone()

        # part(pcb）--------------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.local_conv_list = nn.ModuleList()
        for _ in range(self.parts):
            local_conv = nn.Sequential(
                nn.Conv1d(2048, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )
            self.local_conv_list.append(local_conv)

        self.parts_classifier_list = nn.ModuleList()
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.parts_classifier_list.append(fc)

        # bilstm --------------------------------------------------------------------------
        self.bilstm = nn.LSTM(256, 128, bidirectional=True)
        self.bilstm_classifier_list = nn.ModuleList()
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.bilstm_classifier_list.append(fc)

        # gloab--------------------------------------------------------------------------
        self.k11_conv = nn.Conv2d(2048, 512, kernel_size=1)
        self.gloab_agp = nn.AdaptiveAvgPool2d((1, 1))
        self.gloab_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.gloab_conv.apply(weights_init_kaiming)

        # feature fusion module--------------------------------------------------------------------------
        self.ffm = Feature_Fusion_Module(self.parts)

    def forward(self, x):
        batch_size = x.size(0)

        # backbone(Tensor T)
        resnet_features = self.backbone(x)

        # parts --------------------------------------------------------------------------
        features_G = self.avgpool(resnet_features)  # tensor g([N, 2048, 6, 1])
        features_H = []  # contains 6 ([N, 256, 1])
        for i in range(self.parts):
            stripe_features_H = self.local_conv_list[i](features_G[:, :, i, :])
            features_H.append(stripe_features_H)

        # bilstm --------------------------------------------------------------------------
        features_bilstm = torch.stack(features_H, 0)
        features_bilstm = features_bilstm.squeeze()
        features_bilstm, (_, _) = self.bilstm(features_bilstm)

        # gloab --------------------------------------------------------------------------
        gloab_features = self.k11_conv(resnet_features)
        gloab_features = self.gloab_agp(gloab_features).view(
            batch_size, 512, -1
        )  # ([N, 512, 1])
        gloab_features = self.gloab_conv(gloab_features).squeeze()  # ([N, 512])

        # feature fusion module--------------------------------------------------------------------------
        fusion_feature = self.ffm(gloab_features, features_H)

        if self.training:
            parts_score_list = [
                self.parts_classifier_list[i](features_H[i].view(batch_size, -1))
                for i in range(self.parts)
            ]  # shape list（[N, C=num_classes]）

            lstm_score_list = [
                self.bilstm_classifier_list[i](features_bilstm[i].view(batch_size, -1))
                for i in range(self.parts)
            ]  # shape list（[N, C=num_classes]）

            return parts_score_list, lstm_score_list, gloab_features, fusion_feature
        else:
            v_result = torch.cat(features_H, dim=1)
            v_result = F.normalize(v_result, p=2, dim=1)
            v_result = v_result.view(batch_size, -1)
            return v_result
