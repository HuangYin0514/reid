import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from dataloader.getDataLoader import getData
from evaluators.distance import compute_distance_matrix
from evaluators.feature_extractor import feature_extractor
from evaluators.rank import eval_market1501
from loss.baselineloss import CenterLoss, Softmax_Triplet_loss
from loss.crossEntropyLabelSmoothLoss import CrossEntropyLabelSmoothLoss
from models.baseline_pcb import Baseline_PCB
from optim.WarmupMultiStepLR import WarmupMultiStepLR
from utils import load_network, util
from utils.logger import (Draw_Curve, Logger, print_test_infomation,
                          print_train_infomation)

# from dataloader.utils.RandomErasing import RandomErasing

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# base (env setting)
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
parser.add_argument("--name", type=str, default="person_reid")
# data
parser.add_argument(
    "--data_dir", type=str, default="./datasets/Market-1501-v15.09.15_reduce"
)
# parser.add_a
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# train
parser.add_argument("--num_epochs", type=int, default=2)
# other
parser.add_argument("--img_height", type=int, default=4)
parser.add_argument("--img_width", type=int, default=2)
# print epoch iter
parser.add_argument("--epoch_train_print", type=int, default=1)
parser.add_argument("--epoch_test_print", type=int, default=1)
parser.add_argument("--epoch_start_test", type=int, default=0)


# parse
opt = parser.parse_args()
util.print_options(opt)

# env setting ==============================================================================
# Fix random seed
random_seed = 1
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
# save dir path
save_dir_path = os.path.join(opt.checkpoints_dir, opt.name)
# Logger instance
logger = Logger(save_dir_path)
# draw curve instance
curve = Draw_Curve(save_dir_path)

# data ============================================================================================================
# data Augumentation
train_loader, query_loader, gallery_loader, num_classes = getData(opt)

# model ============================================================================================================
model = Baseline_PCB(num_classes)
model = model.to(device)

# criterion ============================================================================================================
# criterion = F.cross_entropy
# ce_labelsmooth_loss = CrossEntropyLabelSmoothLoss(num_classes=num_classes)
# triplet_loss = TripletLoss(margin=0.3)
use_gpu = False
if device == "cuda":
    use_gpu = True
# criterion = F.cross_entropy
# ce_labelsmooth_loss = CrossEntropyLabelSmoothLoss(num_classes=num_classes)
# triplet_loss = TripletLoss(margin=0.3)

criterion = Softmax_Triplet_loss(
    num_class=num_classes,
    margin=0.3,
    epsilon=0.1,
    use_gpu=use_gpu,
)
center_loss = CenterLoss(
    num_classes=num_classes,
    feature_dim=2048,
    use_gpu=use_gpu,
)
ce_labelsmooth_loss = CrossEntropyLabelSmoothLoss(num_classes=num_classes)

# optimizer ============================================================================================================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.00035,
    weight_decay=0.0005,
)

optimizer_centerloss = torch.optim.SGD(center_loss.parameters(), lr=0.5)

# # scheduler ============================================================================================================
scheduler = WarmupMultiStepLR(
    optimizer,
    milestones=[40, 70],
    gamma=0.1,
    warmup_factor=0.01,
    warmup_iters=10,
    warmup_method="linear",
)

# Training and test ============================================================================================================
def train():
    start_time = time.time()

    for epoch in range(opt.num_epochs):
        model.train()

        running_loss = 0.0
        acc = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # net ---------------------
            optimizer.zero_grad()

            baseline_score, baseline_feat, parts_score_list = model(inputs)

            _, preds = torch.max(baseline_score.data, dim=1)

            loss_baseline = criterion(baseline_score, baseline_feat, labels)
            loss_center = center_loss(baseline_feat, labels) * 0.0005
            loss_part = 0
            for logits in parts_score_list:
                stripe_loss = ce_labelsmooth_loss(logits, labels)
                loss_part += stripe_loss

            loss = loss_baseline + loss_center + loss_part
            loss.backward()

            optimizer.step()
            optimizer_centerloss.step()
            # --------------------------

            running_loss += loss.item() * inputs.size(0)
            acc += torch.sum(preds == labels.data).double().item()
        # scheduler
        scheduler.step()

        if epoch % opt.epoch_train_print == 0:
            accuracy = acc / len(train_loader.dataset) * 100
            print_train_infomation(
                epoch,
                opt.num_epochs,
                running_loss,
                train_loader,
                logger,
                curve,
                start_time,
                accuracy,
            )

        # test
        if epoch % opt.epoch_test_print == 0 and epoch > opt.epoch_start_test:
            # test current datset-------------------------------------
            torch.cuda.empty_cache()
            CMC, mAP = test(epoch)

            print_test_infomation(epoch, CMC, mAP, curve, logger)

    # Save the loss curve
    curve.save_curve()
    # Save final model weights
    load_network.save_network(model, save_dir_path, "final")

    print("training is done !")


@torch.no_grad()
def test(_, normalize_feature=True, dist_metric="cosine"):
    model.eval()

    # Extracting features from query set(matrix size is qf.size(0), qf.size(1))------------------------------------------------------------
    qf, q_pids, q_camids = feature_extractor(query_loader, model, device)
    # print("Done, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    # Extracting features from gallery set(matrix size is gf.size(0), gf.size(1))------------------------------------------------------------
    gf, g_pids, g_camids = feature_extractor(gallery_loader, model, device)

    # normalize_feature------------------------------------------------------------------------------
    if normalize_feature:
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    # Computing distance matrix------------------------------------------------------------------------
    _, rank_results = compute_distance_matrix(qf, gf)

    # Computing CMC and mAP------------------------------------------------------------------------
    CMC, MAP = eval_market1501(rank_results, q_camids, q_pids, g_camids, g_pids)

    return CMC, MAP


if __name__ == "__main__":
    train()