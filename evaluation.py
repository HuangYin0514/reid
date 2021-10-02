import argparse
import os
import random
import time

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from dataloader.collate_batch import val_collate_fn
from dataloader.market1501 import Market1501
from models import *
from utils import reid_util, util, visualize_ranked_results

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# base (env setting)
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
parser.add_argument("--name", type=str, default="person_reid")
# data
parser.add_argument(
    "--data_dir", type=str, default="./datasets/Market-1501-v15.09.15_reduce"
)
# parser.add_argument(
#     "--data_dir", type=str, default="./datasets/Market-1501-v15.09.15"
# )
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# train
parser.add_argument("--num_epochs", type=int, default=2)
# other
parser.add_argument("--img_height", type=int, default=128)
parser.add_argument("--img_width", type=int, default=64)

parser.add_argument(
    "--model_path", type=str, default="./checkpoints/person_reid/net_final-2.pth"
)
# parser.add_argument("--Resize", type=int, default=2)
# parser.add_argument("--CenterCrop", type=int, default=2)

# RandomResizedCrop=224
# Resize=256
# CenterCrop=224

# parse
opt = parser.parse_args()
util.print_options(opt)

# env setting ==============================================================================
# Fix random seed
random_seed = 2021
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)  # Numpy module.
random.seed(random_seed)  # Python random module.
torch.backends.cudnn.deterministic = True
# speed up compution
torch.backends.cudnn.benchmark = True
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("using cuda ...")

save_dir_path = os.path.join(opt.checkpoints_dir, opt.name)

# data ============================================================================================================
test_transforms = transform = T.Compose(
    [
        T.Resize((opt.img_height, opt.img_width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


query_dataset = Market1501(
    root=opt.data_dir, data_folder="query", transform=test_transforms, relabel=False
)
gallery_dataset = Market1501(
    root=opt.data_dir,
    data_folder="bounding_box_test",
    transform=test_transforms,
    relabel=False,
)


def evaluation_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths


query_loader = torch.utils.data.DataLoader(
    query_dataset,
    batch_size=opt.test_batch_size,
    shuffle=False,
    num_workers=opt.num_workers,
    collate_fn=evaluation_collate_fn,
)
gallery_loader = torch.utils.data.DataLoader(
    gallery_dataset,
    batch_size=opt.test_batch_size,
    shuffle=False,
    num_workers=opt.num_workers,
    collate_fn=evaluation_collate_fn,
)

# model ============================================================================================================
model = Resnet_pcb_3branch(1)
model = util.load_network(model, opt.model_path)
model = model.to(device)


def _evaluation_parse_data_for_eval(data):
    imgs = data[0]
    pids = data[1]
    camids = data[2]
    img_paths = data[3]
    return imgs, pids, camids, img_paths


@torch.no_grad()
def test(epoch, normalize_feature=True, dist_metric="cosine"):
    model.eval()

    # Extracting features from query set------------------------------------------------------------
    print("Extracting features from query set ...")
    qf, q_pids, q_camids = (
        [],
        [],
        [],
    )  # query features, query person IDs and query camera IDs
    q_paths = []
    for _, data in enumerate(query_loader):
        imgs, pids, camids, img_paths = _evaluation_parse_data_for_eval(data)
        imgs = imgs.to(device)
        features = reid_util._extract_features(model, imgs)
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
        q_paths.extend(img_paths)
    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Done, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    # Extracting features from gallery set------------------------------------------------------------
    print("Extracting features from gallery set ...")
    gf, g_pids, g_camids = (
        [],
        [],
        [],
    )  # gallery features, gallery person IDs and gallery camera IDs
    g_paths = []
    for _, data in enumerate(gallery_loader):
        imgs, pids, camids, img_paths = _evaluation_parse_data_for_eval(data)
        imgs = imgs.to(device)
        features = reid_util._extract_features(model, imgs)
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
        g_paths.extend(img_paths)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Done, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    # normalize_feature------------------------------------------------------------------------------
    if normalize_feature:
        print("Normalzing features with L2 norm ...")
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    # Computing distance matrix------------------------------------------------------------------------
    print("Computing distance matrix with metric={} ...".format(dist_metric))
    qf = np.array(qf.cpu())
    gf = np.array(gf.cpu())

    # Save to Matlab for check
    result = {
        "gallery_f": gf,
        "gallery_label": g_pids,
        "gallery_cam": g_camids,
        "gallery_path": g_paths,
        "query_f": qf,
        "query_label": q_pids,
        "query_cam": q_camids,
        "query_path": q_paths,
    }
    scipy.io.savemat("pytorch_result.mat", result)
    print("done! save pytorch_result.mat")


if __name__ == "__main__":
    test(1)
