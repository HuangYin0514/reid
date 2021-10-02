import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
from dataloader.market1501 import Market1501
import torchvision.transforms as T
from dataloader.collate_batch import val_collate_fn

matplotlib.use("agg")
import matplotlib.pyplot as plt
from utils import reid_util


def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
parser.add_argument(
    "--query_path_dir", type=str, default="datasets/Market-1501-v15.09.15/query/"
)
parser.add_argument(
    "--gallery_path_dir", type=str, default="datasets/Market-1501-v15.09.15/bounding_box_test/"
)
opt = parser.parse_args()

######################################################################
result = scipy.io.loadmat("checkpoints/person_reid/pytorch_result.mat")
query_feature = torch.FloatTensor(result["query_f"])
query_cam = result["query_cam"][0]
query_label = result["query_label"][0]
query_path = result["query_path"]
gallery_feature = torch.FloatTensor(result["gallery_f"])
gallery_cam = result["gallery_cam"][0]
gallery_label = result["gallery_label"][0]
gallery_path = result["gallery_path"]

# qf = np.array(query_feature.cpu())
# gf = np.array(gallery_feature.cpu())
# dist = reid_util.cosine_dist(qf, gf)
# rank_results = np.argsort(dist)[:, ::-1]
# # Computing CMC and mAP------------------------------------------------------------------------
# print("Computing CMC and mAP ...")
# APs, CMC = [], []
# for _, data in enumerate(zip(rank_results, query_cam, query_label)):
#     a_rank, query_camid, query_pid = data
#     ap, cmc = reid_util.compute_AP(
#         a_rank, query_camid, query_pid, gallery_cam, gallery_label
#     )
#     APs.append(ap), CMC.append(cmc)
# MAP = np.array(APs).mean()
# min_len = min([len(cmc) for cmc in CMC])
# CMC = [cmc[:min_len] for cmc in CMC]
# CMC = np.mean(np.array(CMC), axis=0)
# print(
#     "Testing: top1:%.4f top5:%.4f top10:%.4f mAP:%.4f" % (CMC[0], CMC[4], CMC[9], MAP)
# )


i = 3
# 根据得分，计算得分最高的图像
index = sort_img(
    query_feature[i],
    query_label[i],
    query_cam[i],
    gallery_feature,
    gallery_label,
    gallery_cam,
)


########################################################################
# Visualize the rank result
q_path=opt.query_path_dir+str(query_path[i]).split('/')[-1].replace(" ", "")
query_label = query_label[i]
# print(query_path)
print("Top 10 images are as follow:")
# Visualize Ranking Result
# Graphical User Interface is needed
fig = plt.figure(figsize=(16, 4))
ax = plt.subplot(1, 11, 1)
ax.axis("off")
imshow(q_path, "query")
for i in range(10):
    ax = plt.subplot(1, 11, i + 2)
    ax.axis("off")

    img_path=str(gallery_path[index[i]]).split('/')[-1].replace(" ", "")
    img_path = opt.gallery_path_dir+img_path
    label = gallery_label[index[i]]

    imshow(img_path)
    if label == query_label:
        ax.set_title("{} ".format(label),color="green")
    else:
        ax.set_title("{} ".format(label), color="red")
    print(img_path)

fig.savefig("show.png")
