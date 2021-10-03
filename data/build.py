# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import (
    RandomIdentitySampler,
    RandomIdentitySampler_alignedreid,
)  # New add by gu
from .transforms import build_transforms


def make_data_loader(opt, dataset_name, data_path, sampler="softmax"):
    train_transforms = build_transforms(opt, is_train=True)
    val_transforms = build_transforms(opt, is_train=False)

    dataset = init_dataset(dataset_name, root=data_path)

    num_classes = dataset.num_train_pids
    num_workers = 4
    train_set = ImageDataset(dataset.train, train_transforms)
    if sampler == "softmax":
        train_loader = DataLoader(
            train_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=opt.batch_size,
            sampler=RandomIdentitySampler(
                dataset.train, opt.batch_size, num_instances=4
            ),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers,
            collate_fn=train_collate_fn,
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set,
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )
    return train_loader, val_loader, len(dataset.query), num_classes
