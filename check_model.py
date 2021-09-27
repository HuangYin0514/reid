from models.baseline_pcb import Baseline_PCB
import torch

model = Baseline_PCB(2021)
model.eval()
inputs = torch.randn(4,3,256,128)
outputs = model(inputs)

for output in outputs:
    print(output.shape)
