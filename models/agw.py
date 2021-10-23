# import torch
# import torch.nn as nn
# from .backbones.resnet50 import resnet50
# from .model_utils.init_param import weights_init_classifier, weights_init_kaiming


# class GeneralizedMeanPooling(nn.Module):
#     r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
#     The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
#         - At p = infinity, one gets Max Pooling
#         - At p = 1, one gets Average Pooling
#     The output is of size H x W, for any input size.
#     The number of output features is equal to the number of input planes.
#     Args:
#         output_size: the target output size of the image of the form H x W.
#                      Can be a tuple (H, W) or a single H for a square image H x H
#                      H and W can be either a ``int``, or ``None`` which means the size will
#                      be the same as that of the input.
#     """

#     def __init__(self, norm, output_size=1, eps=1e-6):
#         super(GeneralizedMeanPooling, self).__init__()
#         assert norm > 0
#         self.p = float(norm)
#         self.output_size = output_size
#         self.eps = eps

#     def forward(self, x):
#         x = x.clamp(min=self.eps).pow(self.p)
#         return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(
#             1.0 / self.p
#         )

#     def __repr__(self):
#         return (
#             self.__class__.__name__
#             + "("
#             + str(self.p)
#             + ", "
#             + "output_size="
#             + str(self.output_size)
#             + ")"
#         )


# class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
#     """Same, but norm is trainable"""

#     def __init__(self, norm=3, output_size=1, eps=1e-6):
#         super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
#         self.p = nn.Parameter(torch.ones(1) * norm)


# class Resnet_Backbone(nn.Module):
#     def __init__(self):
#         super(Resnet_Backbone, self).__init__()

#         # backbone--------------------------------------------------------------------------
#         # change the model different from pcb
#         resnet = resnet50(pretrained=True)
#         # Modifiy the stride of last conv layer----------------------------
#         resnet.layer4[0].downsample[0].stride = (1, 1)
#         resnet.layer4[0].conv2.stride = (1, 1)
#         # Remove avgpool and fc layer of resnet------------------------------
#         self.resnet_conv1 = resnet.conv1
#         self.resnet_bn1 = resnet.bn1
#         self.resnet_relu = resnet.relu
#         self.resnet_maxpool = resnet.maxpool
#         self.resnet_layer1 = resnet.layer1
#         self.resnet_layer2 = resnet.layer2
#         self.resnet_layer3 = resnet.layer3
#         self.resnet_layer4 = resnet.layer4

#     def forward(self, x):
#         x = self.resnet_conv1(x)
#         x = self.resnet_bn1(x)
#         x = self.resnet_relu(x)
#         x = self.resnet_maxpool(x)
#         x = self.resnet_layer1(x)
#         x = self.resnet_layer2(x)
#         x = self.resnet_layer3(x)
#         x = self.resnet_layer4(x)

#         return x


# class agw(nn.Module):
#     in_planes = 2048

#     def __init__(self, num_classes):

#         self.num_classes = num_classes

#         super(agw, self).__init__()

#         # backbone
#         self.backbone = Resnet_Backbone()

#         print("Generalized Mean Pooling")
#         self.global_pool = GeneralizedMeanPoolingP()

#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)  # no shift
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)

#     def forward(self, x):
#         batch_size = x.size(0)

#         # backbone(Tensor T)
#         resnet_features = self.backbone(x)

#         global_feat = self.global_pool(resnet_features)  # (b, 2048, 1, 1)
#         global_feat = global_feat.view(
#             global_feat.shape[0], -1
#         )  # flatten to (bs, 2048)

#         feat = self.bottleneck(global_feat)  # normalize for angular softmax

#         ######################################################################################################################
#         # Return the features_H if inference--------------------------------------------------------------------------
#         if not self.training:

#             return x
#         else:

#             return x
