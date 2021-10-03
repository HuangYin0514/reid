import torch.nn as nn
import torch


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1
        )  # one-hot encoding
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(dim=0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # ||a-b||^2 = ||a||^2 -2 * <a,b> + ||b||^2
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist = torch.addmm(input=dist, mat1=inputs, mat2=inputs.t(), alpha=-2, beta=1)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][mask[i] == True].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == False].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)

        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        return self.ranking_loss(dist_an, dist_ap, y)


class Softmax_Triplet_loss(nn.Module):
    def __init__(self, num_class, margin, epsilon, use_gpu):
        super().__init__()
        self.cross_entropy = CrossEntropyLabelSmooth(
            num_classes=num_class, epsilon=epsilon, use_gpu=use_gpu
        )
        self.triplet = TripletLoss(margin=margin)

    def forward(self, score, feat, target):
        return self.cross_entropy(score, target) + self.triplet(feat, target)


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    https://github.com/KaiyangZhou/pytorch-center-loss
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feature_dim, use_gpu):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feature_dim).cuda()
            )
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        # distmat.addmm_(1, -2, x, self.centers.t())
        distmat = torch.addmm(
            input=distmat, mat1=x, mat2=self.centers.t(), alpha=-2, beta=1
        )

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size
        return loss
