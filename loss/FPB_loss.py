import torch.nn as nn
import torch
import warnings

# devie-------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.
    
    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    With label smoothing, the label :math:`y` for a class is computed by
    
    .. math::
        \begin{equation}
        (1 - \epsilon) \times y + \frac{\epsilon}{K},
        \end{equation}
    where :math:`K` denotes the number of classes and :math:`\epsilon` is a weight. When
    :math:`\epsilon = 0`, the loss function reduces to the normal cross entropy.
    
    Args:
        num_classes (int): number of classes.
        epsilon (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """
    
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.to(device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return (- targets * log_probs).mean(0).sum()


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
        if margin == 0.:
            self.ranking_loss = nn.SoftMarginLoss()
        else:        
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        if self.margin == 0.:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
    - num_classes (int): number of classes.
    - feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        warnings.warn("This method is deprecated")
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
        - x: feature matrix with shape (batch_size, feat_dim).
        - labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.to(device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss