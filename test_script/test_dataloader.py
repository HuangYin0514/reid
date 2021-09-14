
from torch.utils import data
from dataloader import hymenoptera

dataset = hymenoptera.Hymenoptera('datasets/hymenoptera_data',mode="train")
dataset = hymenoptera.Hymenoptera('datasets/hymenoptera_data',mode="val")

print(dataset)