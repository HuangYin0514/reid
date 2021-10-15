import argparse
import scipy.io
import torch
import matplotlib.pyplot as plt
from utils.mistake_utils import parse_data, sort_img, visualize_rank_result

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
parser.add_argument(
    "--query_path_dir",
    type=str,
    default="/Users/huangyin/Documents/datasets/Market-1501-v15.09.15/query/",
)
parser.add_argument(
    "--gallery_path_dir",
    type=str,
    default="/Users/huangyin/Documents/datasets/Market-1501-v15.09.15/bounding_box_test/",
)
parser.add_argument(
    "--loadmat_dir",
    type=str,
    default="checkpoints/person_reid/pytorch_result.mat",
)
opt = parser.parse_args()

######################################################################
if __name__ == "__main__":

    result = scipy.io.loadmat(opt.loadmat_dir)

    query_feature, query_cam, query_label, query_path = parse_data(
        result, parse_name="query"
    )

    gallery_feature, gallery_cam, gallery_label, gallery_path = parse_data(
        result, parse_name="gallery"
    )
    for img_index in range(len(query_feature)):

        # 根据得分，计算得分最高的图像
        sort_index = sort_img(
            query_feature[img_index],
            query_label[img_index],
            query_cam[img_index],
            gallery_feature,
            gallery_label,
            gallery_cam,
        )

        # Visualize the rank result
        visualize_rank_result(
            opt, query_path, query_label, gallery_path, gallery_label, sort_index, img_index
        )
