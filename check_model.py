from models.asnet import asnet
import torch

model = asnet(2021)

inputs = torch.randn(4,3,384,192)
outputs = model(inputs)

for output in outputs:
    print(output.shape)
