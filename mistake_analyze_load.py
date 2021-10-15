import argparse
import torch
import torch.nn.functional as F
from data.getDataLoader import getData
from evaluators import distance, feature_extractor, rank
from models.baseline_apne_drop import baseline_apne_drop
from utils import network_module

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

device = "cuda" if torch.cuda.is_available() else "cpu"

# data ============================================================================================================
train_loader, query_loader, gallery_loader, num_classes = getData(opt)
# model ============================================================================================================
model = baseline_apne_drop(num_classes)
model = model.to(device)
network_module.load_network(model, opt.pretrain_dir)

@torch.no_grad()
def produce_mat(q_loader, g_loader, normalize_feature=True):
    model.eval()

    # Extracting features from query set(matrix size is qf.size(0), qf.size(1))------------------------------------------------------------
    qf, q_pids, q_camids = feature_extractor.feature_extract(q_loader, model, device)

    # Extracting features from gallery set(matrix size is gf.size(0), gf.size(1))------------------------------------------------------------
    gf, g_pids, g_camids = feature_extractor.feature_extract(g_loader, model, device)

    # normalize_feature------------------------------------------------------------------------------
    if normalize_feature:
        # print("Normalzing features with L2 norm ...")
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    # Computing distance matrix------------------------------------------------------------------------
    _, rank_results = distance.compute_distance_matrix(qf, gf)

    # Computing CMC and mAP------------------------------------------------------------------------
    CMC, MAP = rank.eval_market1501(rank_results, q_camids, q_pids, g_camids, g_pids)

    return CMC, MAP

    
######################################################################
if __name__ == "__main__":
    produce_mat()
