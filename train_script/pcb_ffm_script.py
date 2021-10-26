import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from data.getDataLoader import getData
from data.getDataLoader_OccludedREID import getOccludedData
from loss.pcb_loss import CrossEntropyLabelSmoothLoss, TripletLoss
from models.pcb_ffm import pcb_ffm
from utils import network_module
from utils.draw_curve import Draw_Curve
from utils.logger import Logger
from utils.print_infomation import (
    print_options,
    print_test_infomation,
    print_train_infomation,
)

from .test_model import test

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# base (env setting)
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
parser.add_argument("--name", type=str, default="person_reid")
# data
parser.add_argument(
    "--data_dir",
    type=str,
    default="/Users/huangyin/Documents/datasets/Market-1501-v15.09.15_reduce",
)
parser.add_argument(
    "--test_data_dir",
    type=str,
    default="/Users/huangyin/Documents/datasets/Occluded_REID_reduce",
)
parser.add_argument("--batch_size", default=50, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--img_height", type=int, default=384)
parser.add_argument("--img_width", type=int, default=128)
# train
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--pretrain_dir", type=str, default="checkpoints/person_reid/")
# print epoch iter
parser.add_argument("--epoch_train_print", type=int, default=1)
parser.add_argument("--epoch_test_print", type=int, default=1)
parser.add_argument("--epoch_start_test", type=int, default=0)


# parse
opt = parser.parse_args()
print_options(opt)

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
query_occluded_loader, gallery_occluded_loader = getOccludedData(
    opt, data_dir=opt.test_data_dir
)

# model ============================================================================================================
model = pcb_ffm(num_classes)
model = model.to(device)
network_module.load_network(model, opt.pretrain_dir)


# criterion ============================================================================================================
use_gpu = False
if device == "cuda":
    use_gpu = True
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

            # model-------------------------------------------------
            parts_scores, gloab_features, fusion_feature = model(inputs)

            ####################################################################
            # gloab loss-------------------------------------------------
            gloab_loss = triplet_loss(gloab_features, labels)

            # fusion loss-------------------------------------------------
            fusion_loss = triplet_loss(fusion_feature, labels)

            # parts loss-------------------------------------------------
            part_loss = 0
            for logits in parts_scores:
                stripe_loss = ce_labelsmooth_loss(logits, labels)
                part_loss += stripe_loss

            # all of loss -------------------------------------------------
            loss_alph = 1
            loss_beta = 0.015
            loss = (
                0.1 * part_loss + loss_alph * gloab_loss[0] + loss_beta * fusion_loss[0]
            )

            loss.backward()

            optimizer.step()
            # --------------------------

            running_loss += loss.item() * inputs.size(0)

        # scheduler
        scheduler.step()

        if epoch % opt.epoch_train_print == 0:
            print_train_infomation(
                epoch,
                opt.num_epochs,
                running_loss,
                train_loader,
                logger,
                curve,
                start_time,
            )

        # test
        if epoch == 0 or (
            epoch % opt.epoch_test_print == 0 and epoch > opt.epoch_start_test
        ):
            # test current datset
            torch.cuda.empty_cache()
            CMC, mAP = test(query_loader, gallery_loader, model)
            print_test_infomation(epoch, CMC, mAP, curve, logger, pattern="ori_dataset")

            # test other datset
            torch.cuda.empty_cache()
            CMC, mAP = test(query_occluded_loader, gallery_occluded_loader, model)
            print_test_infomation(
                epoch, CMC, mAP, curve, logger, pattern="dest_dataset"
            )

    # Save the loss curve
    curve.save_curve()
    # Save final model weights
    network_module.save_network(model, save_dir_path, "final")

    print("training is done !")
