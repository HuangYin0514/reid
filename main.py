import argparse
import os

import torch
from torch.backends import cudnn

import dataloader
from dataloader import getDataLoader
from loss.crossEntropyLabelSmoothLoss import CrossEntropyLabelSmoothLoss
from loss.TripleLoss import TripletLoss
from models import build_model
from train_model import train

parser = argparse.ArgumentParser(description="Person ReID Frame")

# System parameters#System parameters-------------------------------------------------------------
parser.add_argument(
    "--nThread", type=int, default=4, help="number of threads for data loading"
)
parser.add_argument("--nGPU", type=int, default=1, help="number of GPUs")
parser.add_argument("--save_path", type=str, default="../experiments")
parser.add_argument("--experiment", type=str, default="pcb_gloab_ffm")

# Data parameters-------------------------------------------------------------
parser.add_argument("--dataset_name", type=str, default="market1501")
parser.add_argument(
    "--dataset_path", type=str, default="/home/hy/vscode/data/Market-1501-v15.09.15"
)
parser.add_argument(
    "--img_height", type=int, default=2, help="height of the input image"
)
parser.add_argument(
    "--img_width", type=int, default=1, help="width of the input image"
)
parser.add_argument("--batch_size", default=6, type=int, help="batch_size")
parser.add_argument("--test_batch_size", default=6, type=int, help="test_batch_size")
parser.add_argument("--data_sampler_type", type=str, default="RandomIdentitySampler")
parser.add_argument("--num_instance", type=int, default=2)


# Model parameters-------------------------------------------------------------
parser.add_argument("--stripes", type=int, default=6)


# Train parameters-------------------------------------------------------------
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--test_every", type=int, default=2)
parser.add_argument("--fixbase_epoch", type=int, default=0)


# Optimizer parameters-------------------------------------------------------------
parser.add_argument("--lr", type=float, default=0.1)


# Learning rate parameters-------------------------------------------------------------
parser.add_argument("--decay_every", type=int, default=20)
parser.add_argument("--gamma", type=float, default=0.1)

# test other datset parameters-------------------------------------------------------------
parser.add_argument("--test_other_dataset_name", type=str, default="Paritial_REID")
parser.add_argument(
    "--test_other_dataset_path",
    type=str,
    default="/home/hy/vscode/data/Partial-REID_Dataset",
)

args = parser.parse_args()


if __name__ == "__main__":
    # ============================================================================================================
    # devie-------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seed---------------------------------------------------------------------------
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # speed up compution---------------------------------------------------------------------------
    cudnn.benchmark = True

    # ============================================================================================================
    # data------------------------------------------------------------------------------------
    train_loader, query_loader, gallery_loader, num_classes = getDataLoader(
        args.dataset_name, args.dataset_path, args=args
    )
    (
        test_loader,
        test_query_loader,
        test_gallery_loader,
        test_num_classes,
    ) = getDataLoader(
        args.test_other_dataset_name, args.test_other_dataset_path, args=args
    )

    train_data_loader = [train_loader, query_loader, gallery_loader]
    test_data_loader = [test_query_loader, test_gallery_loader]

    dataloader = [train_data_loader, test_data_loader]

    # ============================================================================================================
    # model------------------------------------------------------------------------------------
    model = build_model(num_classes=num_classes, args=args)
    model = model.to(device)

    # criterion-----------------------------------------------------------------------------------
    ce_labelsmooth_loss = CrossEntropyLabelSmoothLoss(num_classes=num_classes)
    MARGIN = 0.3
    triplet_loss = TripletLoss(margin=MARGIN)
    criterion = [ce_labelsmooth_loss, triplet_loss]

    # optimizer-----------------------------------------------------------------------------------
    base_param_ids = set(map(id, model.backbone.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.lr / 10},
        {"params": new_params, "lr": args.lr},
    ]
    optimizer = torch.optim.SGD(
        param_groups, momentum=0.9, weight_decay=5e-4, nesterov=True
    )

    # scheduler-----------------------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ============================================================================================================
    # save_dir_path-----------------------------------------------------------------------------------
    save_dir_path = os.path.join(args.save_path, args.dataset_name)
    os.makedirs(save_dir_path, exist_ok=True)

    # ============================================================================================================
    # train -----------------------------------------------------------------------------------
    train(
        model, criterion, optimizer, scheduler, dataloader, device, save_dir_path, args
    )
