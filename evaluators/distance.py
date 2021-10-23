import numpy as np
import torch


def cosine_dist(x, y):
    """compute cosine distance between two martrix x and y with sizes (n1, d), (n2, d)"""

    def normalize(x):
        """normalize a 2d matrix along axis 1"""
        norm = np.tile(
            np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]]
        )
        return x / norm

    x = normalize(x)
    y = normalize(y)
    return np.matmul(x, y.transpose([1, 0]))


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    distmat = (
        torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat


def compute_distance_matrix(qf, gf, dist_metric="euclidean_squared_distance"):
    if dist_metric == "cosine_dist":
        qf = np.array(qf.cpu())
        gf = np.array(gf.cpu())
        dist = cosine_dist(qf, gf)
        rank_results = np.argsort(dist)[:, ::-1]
    if dist_metric == "euclidean_squared_distance":
        dist = euclidean_squared_distance(qf, gf) #（bs_q,bs_g)
        dist = dist.cpu() #（bs_q,bs_g) sorted by distance
        rank_results = np.argsort(dist, axis=1)
        
    return dist, rank_results
