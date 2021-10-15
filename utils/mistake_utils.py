import torch
import numpy as np
import matplotlib
from utils.common import mkdirs

matplotlib.use("agg")
import matplotlib.pyplot as plt


def parse_data(data, parse_name):

    feature = torch.FloatTensor(data[parse_name + "_f"])
    cam = data[parse_name + "_cam"][0]
    label = data[parse_name + "_label"][0]
    path = data[parse_name + "_path"]
    return feature, cam, label, path


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
    # plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_rank_result(
    opt, query_path, query_label, gallery_path, gallery_label, sort_index, img_index
):
    save_path = "checkpoints/person_reid/mistake/"
    mkdirs(save_path)

    q_path = opt.query_path_dir + str(query_path[img_index]).split("/")[-1].replace(
        " ", ""
    )
    query_label = query_label[img_index]

    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 11, 1)
    ax.axis("off")
    imshow(q_path, "q_" + str(query_label))

    save_flag = False
    for i in range(10):
        ax = plt.subplot(1, 11, i + 2)
        ax.axis("off")

        img_path = str(gallery_path[sort_index[i]]).split("/")[-1].replace(" ", "")
        img_path = opt.gallery_path_dir + img_path
        imshow(img_path)

        label = gallery_label[sort_index[i]]
        if label == query_label:
            ax.set_title("{} ".format(label), color="green")
        else:
            ax.set_title("{} ".format(label), color="red")

        if i == 0 and label != query_label:
            save_flag = True

    if save_flag:
        fig.savefig(save_path + str(query_label) + ".png")
