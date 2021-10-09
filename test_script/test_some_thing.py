import torch
import torch.nn.functional as F
from torch import nn


class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()

        self.nums = 3
        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(
                nn.Conv2d(
                    448,
                    448,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            bns.append(nn.BatchNorm2d(448))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        spx = torch.split(x, 448, dim=1)
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

        ms = torch.split(out, 448, dim=1)
        m12 = torch.cat([ms[0], ms[1]], dim=1)
        m34 = torch.cat([ms[2], ms[3]], dim=1)

        m = [m12,m34]

        return m


if __name__ == "__main__":
    db2d = test_model()
    inputs = torch.randn(32, 1792, 6, 1)
    outputs = db2d(inputs)
    # print(outputs.shape)
