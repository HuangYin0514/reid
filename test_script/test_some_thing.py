import torch
import torch.nn.functional as F
from torch import nn


class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()

        width = 3
        self.nums = 3

        self.width = width

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.stype = "normal"
        self.relu = nn.ReLU(inplace=True)

    def forward(self, y1, y2, y3, y4):
        spx = [y1, y2, y3, y4]
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

        return out


if __name__ == "__main__":
    db2d = test_model()
    inputs = torch.randn(32, 3, 43, 32)
    outputs = db2d(inputs, inputs, inputs, inputs)
    print(outputs.shape)
