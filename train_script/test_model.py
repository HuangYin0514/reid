import torch
from evaluators import distance, feature_extractor, rank
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def test(q_loader, g_loader, model, normalize_feature=True):
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
