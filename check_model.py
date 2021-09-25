from models.Resnet_strongbaseline import StrongBaseline
import torch

model = StrongBaseline(2021)

inputs = torch.randn(4,3,256,128)
outputs = model(inputs)

for output in outputs:
    print(output.shape)
