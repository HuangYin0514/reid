import argparse

import matplotlib
import numpy as np
import torch

from data.getDataLoader import getData
from utils.common import mkdirs

matplotlib.use("agg")
import matplotlib.pyplot as plt

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# data
parser.add_argument(
    "--data_dir",
    type=str,
    default="/Users/huangyin/Documents/datasets/Market-1501-v15.09.15_reduce",
)
parser.add_argument(
    "--test_data_dir", type=str, default="./datasets/Occluded_REID_reduce"
)
# parser.add_a
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--img_height", type=int, default=256)
parser.add_argument("--img_width", type=int, default=128)
opt = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)


def image_tensor_to_numpy(image):
    img = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    return img


if __name__ == "__main__":
    train_loader, query_loader, gallery_loader, num_classes = getData(opt)

    t_imgs, _ = next(iter(train_loader))
    q_imgs, _, _ = next(iter(query_loader))

    fig = plt.figure()
    for i in range(6):
        plt.subplot(4, 3, i + 1)
        plt.axis("off")
        img = t_imgs[i]
        img_ = image_tensor_to_numpy(img)
        plt.imshow(img_, interpolation="none")

    for i in range(6, 12):
        plt.subplot(4, 3, i + 1)
        plt.axis("off")
        img = q_imgs[i]
        img_ = image_tensor_to_numpy(img)
        plt.imshow(img_, interpolation="none")

    save_path = "checkpoints/person_reid/check_data/"
    mkdirs(save_path)    
    fig.savefig(save_path + "checkdata.png")


