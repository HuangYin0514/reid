#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :test_data.py
@说明        :测试partialdataset数据集
@时间        :2020/11/23 11:46:07
@作者        :HuangYin
@版本        :1.0
'''
from dataloader import getDataLoader
import argparse

parser = argparse.ArgumentParser(description='Person ReID Frame')

# Data parameters-------------------------------------------------------------
parser.add_argument('--img_height', type=int, default=384, help='height of the input image')
parser.add_argument('--img_width', type=int, default=128, help='width of the input image')
parser.add_argument('--batch_size', default=6, type=int, help='batch_size')
parser.add_argument('--test_batch_size', default=6, type=int, help='test_batch_size')
parser.add_argument('--data_sampler_type', type=str, default='RandomIdentitySampler')
parser.add_argument('--num_instance', type=int, default=2)

args = parser.parse_args()
dataset_name = 'Paritial_REID'
dataset_path = '/home/hy/vscode/data/Partial-REID_Dataset'

train_loader, query_loader, gallery_loader, num_classes = getDataLoader(dataset_name, dataset_path, args=args)
train_loader
