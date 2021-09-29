import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from dataloader.getPCBDataLoader import getData
from evaluators.distance import compute_distance_matrix
from evaluators.feature_extractor import feature_extractor
from evaluators.rank import eval_market1501
from loss.crossEntropyLabelSmoothLoss import CrossEntropyLabelSmoothLoss
from loss.TripleLoss import TripletLoss
from models.pcb_bilstm_3branch import pcb_bilstm_3branch
from utils import load_network, util
from utils.logger import (
    Draw_Curve,
    Logger,
    print_test_infomation,
    print_train_infomation,
)

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# base (env setting)
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
parser.add_argument("--name", type=str, default="person_reid")
# data
parser.add_argument(
    "--data_dir", type=str, default="./datasets/Market-1501-v15.09.15_reduce"
)
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# train
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--pretrain_dir", type=str, default="checkpoints/person_reid/")

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
model = pcb_bilstm_3branch(num_classes)
model = model.to(device)
load_network.load_network(model, opt.pretrain_dir)


# criterion ============================================================================================================
criterion = F.cross_entropy
ce_labelsmooth_loss = CrossEntropyLabelSmoothLoss(num_classes=num_classes)
triplet_loss = TripletLoss(margin=0.3)

# optimizer ============================================================================================================
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr = 0.1
base_param_ids = set(map(id, model.backbone.parameters()))
new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
param_groups = [
    {"params": model.backbone.parameters(), "lr": lr / 100},
    {"params": new_params, "lr": lr},
]
optimizer = torch.optim.SGD(
    param_groups, momentum=0.9, weight_decay=5e-4, nesterov=True
)

# # scheduler ============================================================================================================
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# Training and test ============================================================================================================
def train():
    start_time = time.time()

    for epoch in range(opt.num_epochs):
        model.train()

        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # net ---------------------
            optimizer.zero_grad()

            parts_scores, lstm_score, gloab_features, fusion_feature = model(inputs)

            # parts loss -------------------------------------------------
            part_loss = 0
            for logits in parts_scores:
                stripe_loss = ce_labelsmooth_loss(logits, labels)
                part_loss += stripe_loss

            # lstm loss -------------------------------------------------
            lstm_loss = 0
            for logits in lstm_score:
                stripe_loss = ce_labelsmooth_loss(logits, labels)
                lstm_loss += stripe_loss

            # gloab loss-------------------------------------------------
            gloab_loss = triplet_loss(gloab_features, labels)

            # fusion loss-------------------------------------------------
            fusion_loss = triplet_loss(fusion_feature, labels)

            loss = part_loss + lstm_loss + 0.15*gloab_loss[0]+ 0*fusion_loss[0]

            loss.backward()
            optimizer.step()
            # --------------------------

            running_loss += loss.item() * inputs.size(0)

        # scheduler
        scheduler.step()

        if epoch % opt.epoch_train_print == 0:
            accuracy = 0
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
