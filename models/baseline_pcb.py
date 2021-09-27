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


class Baseline_PCB(nn.Module):
    def __init__(self, num_classes):

        self.num_classes = num_classes

        super(Baseline_PCB, self).__init__()

        # backbone
        self.backbone = Resnet_Backbone()

        # baseline
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        # part
        self.parts = 6
        self.part_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.local_conv_list = nn.ModuleList()
        for _ in range(self.parts):
            local_conv = nn.Sequential(
                nn.Conv1d(2048, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )
            self.local_conv_list.append(local_conv)
        self.parts_classifier_list = (
            nn.ModuleList()
        )  # Classifier for each stripe （parts feature）
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.parts_classifier_list.append(fc)

    def forward(self, x):
        batch_size = x.size(0)

        resnet_features = self.backbone(x)  # (batch_size, 2048, 16, 8)

        # baseline
        baseline_out = self.avgpool(resnet_features)  # (batch_size, 2048, 1, 1)
        baseline_out = baseline_out.view(batch_size, -1)  # (batch_size, 2048)
        baseline_feat = self.bottleneck(baseline_out)  # (batch_size, 2048)

        features_G = self.part_avgpool(
            resnet_features
        )  # tensor g([batch_size, 2048, 6, 1])
        features_H = []  # contains 6 ([batch_size, 256, 1])
        for i in range(self.parts):
            stripe_features_H = self.local_conv_list[i](features_G[:, :, i, :])
            features_H.append(stripe_features_H)

        if self.training:
            baseline_score = self.classifier(baseline_feat)  # (batch_size, num_classes)
            parts_score_list = [
                self.parts_classifier_list[i](features_H[i].view(batch_size, -1))
                for i in range(self.parts)
            ]  # （batch_size, num_classes）
            return baseline_score, baseline_feat, parts_score_list
        else:
            # inference

            # baseline
            v_baseline = baseline_feat #（batch_size, 2048)
            # part
            v_part = torch.cat(features_H, dim=1)  #（batch_size, 1536)
            # v_part = F.normalize(v_part, p=2, dim=1)
            v_part = v_part.view(batch_size, -1)

            #result
            v_result = torch.cat([v_baseline,v_baseline],dim=1)
            v_result = F.normalize(v_result, p=2, dim=1)
            v_result = v_result.view(batch_size, -1)

            return v_result
