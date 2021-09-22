from models.APNet import Baseline
import torch

model = Baseline(1024, level=2)

inputs = torch.randn(4,3,384,192)
outputs = model(inputs)

for output in outputs:
    print(output.shape)
