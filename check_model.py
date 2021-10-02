import torch

from models.pcb_bilstm_3branch import pcb_bilstm_3branch
model = pcb_bilstm_3branch(2021)

# from models.baseline import Baseline
# model = Baseline(2021)

inputs = torch.randn(6, 3, 256, 128)


def train():
    print("this is a train method")
    model.train()
    outputs = model(inputs)
    for i, output in enumerate(outputs):
        if isinstance(output, list):
            for j, o in enumerate(output):
                print("{}-{} banch is {}".format(i, j, o.shape))
        else:
            print("{} banch is {}".format(i, output.shape))


def test():
    print("this is a test method")
    model.eval()
    outputs = model(inputs)
    if isinstance(outputs, list):
        for i, o in enumerate(outputs):
            print("{} banch is {}".format(i, o.shape))
    else:
        print("outputs is {}".format(outputs.shape))


train()
test()
