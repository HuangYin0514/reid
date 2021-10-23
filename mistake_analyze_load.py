import argparse
import torch
import torch.nn.functional as F
from data.getDataLoader_mistake import getData
from evaluators import distance, rank
from models.baseline_apne_drop import baseline_apne_drop
from utils import network_module
import scipy.io
import numpy as np
from utils.common import mkdirs

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

# data
parser.add_argument(
    "--data_dir",
    type=str,
    default="/Users/huangyin/Documents/datasets/Market-1501-v15.09.15_reduce",
)
parser.add_argument(
    "--test_data_dir", type=str, default="./datasets/Occluded_REID_reduce"
)

parser.add_argument("--batch_size", default=50, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--img_height", type=int, default=2)
parser.add_argument("--img_width", type=int, default=1)

parser.add_argument("--pretrain_dir", type=str, default="checkpoints/person_reid/")

opt = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# data ============================================================================================================
train_loader, query_loader, gallery_loader, num_classes = getData(opt)
# model ============================================================================================================
model = baseline_apne_drop(num_classes)
model = model.to(device)
network_module.load_network(model, opt.pretrain_dir)


def _evaluation_parse_data_for_eval(data):
    imgs = data[0]
    pids = data[1]
    camids = data[2]
    img_paths = data[3]
    return imgs, pids, camids, img_paths


def _extract_features(model, input):
    model.eval()
    return model(input)


@torch.no_grad()
def produce_mat(q_loader, g_loader, normalize_feature=True):
    model.eval()

    model.eval()

    # Extracting features from query set------------------------------------------------------------
    print("Extracting features from query set ...")
    qf, q_pids, q_camids = (
        [],
        [],
        [],
    )  # query features, query person IDs and query camera IDs
    q_paths = []

    for _, data in enumerate(q_loader):
        imgs, pids, camids, img_paths = _evaluation_parse_data_for_eval(data)
        imgs = imgs.to(device)
        features = _extract_features(model, imgs)
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
    for _, data in enumerate(g_loader):
        imgs, pids, camids, img_paths = _evaluation_parse_data_for_eval(data)
        imgs = imgs.to(device)
        features = _extract_features(model, imgs)
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

    qf_cpu = np.array(qf.cpu())
    gf_cpu = np.array(gf.cpu())

    # Save to Matlab for check
    result = {
        "gallery_f": gf_cpu,
        "gallery_label": g_pids,
        "gallery_cam": g_camids,
        "gallery_path": g_paths,
        "query_f": qf_cpu,
        "query_label": q_pids,
        "query_cam": q_camids,
        "query_path": q_paths,
    }

    mkdirs("checkpoints/person_reid/")
    scipy.io.savemat("checkpoints/person_reid/pytorch_result.mat", result)
    print("done! save pytorch_result.mat")

  


    # Computing distance matrix------------------------------------------------------------------------
    _, rank_results = distance.compute_distance_matrix(qf, gf)

    # Computing CMC and mAP------------------------------------------------------------------------
    CMC, MAP = rank.eval_market1501(rank_results, q_camids, q_pids, g_camids, g_pids)

    return CMC, MAP


######################################################################
if __name__ == "__main__":

    CMC, mAP = produce_mat(query_loader, gallery_loader)

    print(
        "Testing: top1:%.4f top5:%.4f top10:%.4f mAP:%.4f"
        % (CMC[0], CMC[4], CMC[9], mAP)
    )
