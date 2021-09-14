import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Resnet18Custom(nn.Module):
    def __init__(self):
        super(Resnet18Custom, self).__init__()

        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        x = self.model(x)
        return x
