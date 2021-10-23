import argparse
import os
import random
import time

import numpy as np
import torch
from data.getDataLoader import getData
from data.getDataLoader_OccludedREID import getOccludedData
from loss.baseline_loss import CenterLoss, Softmax_Triplet_loss
from models.baseline_apne_drop import baseline_apne_drop
from optim.WarmupMultiStepLR import WarmupMultiStepLR
from utils import network_module
from utils.draw_curve import Draw_Curve
from utils.logger import Logger
from .test_model import test
from utils.print_infomation import (
    print_options,
    print_test_infomation,
    print_train_infomation,
)

from loss.pcb_loss import CrossEntropyLabelSmoothLoss

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
    "--test_data_dir", type=str, default="/Users/huangyin/Documents/datasets/Occluded_REID_reduce"
)
parser.add_argument("--batch_size", default=50, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--img_height", type=int, default=2)
parser.add_argument("--img_width", type=int, default=1)
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
model = baseline_apne_drop(num_classes)
model = model.to(device)
network_module.load_network(model, opt.pretrain_dir)


# criterion ============================================================================================================
use_gpu = False
if device == "cuda":
    use_gpu = True


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
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # net ---------------------
            optimizer.zero_grad()

            score, feat, parts_score_list, parts_score_list2, parts_score_list3 = model(
                inputs
            )

            # parts loss-------------------------------------------------
            part_loss = 0
            for logits in parts_score_list:
                stripe_loss = ce_labelsmooth_loss(logits, labels)
                part_loss += stripe_loss

            part_loss2 = 0
            for logits in parts_score_list2:
                stripe_loss = ce_labelsmooth_loss(logits, labels)
                part_loss2 += stripe_loss

            part_loss3 = 0
            for logits in parts_score_list3:
                stripe_loss = ce_labelsmooth_loss(logits, labels)
                part_loss3 += stripe_loss

            loss = (
                criterion(score, feat, labels)
                + center_loss(feat, labels) * 0.0005
                + part_loss * 0.01
                + part_loss2 * 0.01
                + part_loss3 * 0.01
            )

            loss.backward()

            optimizer.step()
            optimizer_centerloss.step()
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
            CMC, mAP = test(query_loader, gallery_loader,model)
            print_test_infomation(epoch, CMC, mAP, curve, logger, pattern="ori_dataset")

            # test other datset
            torch.cuda.empty_cache()
            CMC, mAP = test(query_occluded_loader, gallery_occluded_loader,model)
            print_test_infomation(
                epoch, CMC, mAP, curve, logger, pattern="dest_dataset"
            )


    # Save the loss curve
    curve.save_curve()
    # Save final model weights
    network_module.save_network(model, save_dir_path, "final")

    print("training is done !")
