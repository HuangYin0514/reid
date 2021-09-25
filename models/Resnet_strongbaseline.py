
import torch
import torch.nn as nn
from .model_utils.resnet50 import resnet50
from .model_utils.init_param import weights_init_classifier,weights_init_kaiming


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


class StrongBaseline(nn.Module):
    def __init__(self, num_classes):

        self.parts = 6
        self.in_planes = 2048
        
        super(StrongBaseline, self).__init__()

        # backbone
        self.backbone = Resnet_Backbone()

        # # part(pcb）--------------------------------------------------------------------------
        # self.avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        # self.local_conv_list = nn.ModuleList()
        # for _ in range(self.parts):
        #     local_conv = nn.Sequential(
        #         nn.Conv1d(2048, 256, kernel_size=1),
        #         nn.BatchNorm1d(256),
        #         nn.ReLU(inplace=True),
        #     )
        #     self.local_conv_list.append(local_conv)

        # # Classifier for each stripe （parts feature）-------------------------------------
        # self.parts_classifier_list = nn.ModuleList()
        # for _ in range(self.parts):
        #     fc = nn.Linear(256, num_classes)
        #     nn.init.normal_(fc.weight, std=0.001)
        #     nn.init.constant_(fc.bias, 0)
        #     self.parts_classifier_list.append(fc)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck_feat = 'after'

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        batch_size = x.size(0)

        # # backbone(Tensor T)
        # resnet_features = self.backbone(x)

        # # parts --------------------------------------------------------------------------
        # features_G = self.avgpool(resnet_features)  # tensor g([N, 2048, 6, 1])
        # features_H = []  # contains 6 ([N, 256, 1])
        # for i in range(self.parts):
        #     stripe_features_H = self.local_conv_list[i](features_G[:, :, i, :])
        #     features_H.append(stripe_features_H)

        # ######################################################################################################################
        # # Return the features_H if inference--------------------------------------------------------------------------
        # if not self.training:
        #     # features_H.append(gloab_features.unsqueeze_(2))  # ([N,1536+512])
        #     v_g = torch.cat(features_H, dim=1)
        #     v_g = F.normalize(v_g, p=2, dim=1)
        #     return v_g.view(v_g.size(0), -1)
        # else:
        #     ######################################################################################################################
        #     # classifier(parts)--------------------------------------------------------------------------
        #     parts_score_list = [
        #         self.parts_classifier_list[i](features_H[i].view(batch_size, -1))
        #         for i in range(self.parts)
        #     ]  # shape list（[N, C=num_classes]）

        # return parts_score_list

        global_feat = self.gap(self.backbone(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(
            global_feat.shape[0], -1
        )  # flatten to (bs, 2048)

        
        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            # print("Test with feature after BN")
            return feat
           
if __name__ == '__main__':
    model = StrongBaseline(2021)

    inputs = torch.randn(4,3,256,128)
    outputs = model(inputs)

    for output in outputs:
        print(output.shape)

    