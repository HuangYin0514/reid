from models import *
import torch

model = Resnet_pcb_2branch(10)

inputs = torch.randn(20,3,384,128)
outputs = model(inputs)

for output in outputs:
    print(output.shape)