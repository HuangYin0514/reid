#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :partial_reid.py
@说明        :Partial-REID_Dataset
@时间        :2020/11/23 10:49:50
@作者        :HuangYin
@版本        :1.0
'''

import glob
import os.path as osp
from .bases import BaseImageDataset


class Paritial_REID(BaseImageDataset):

    dataset_dir = ''

    def __init__(self, root='', verbose=True, **kwargs):
        super(Paritial_REID, self).__init__()

        self.dataset_dir = osp.abspath(osp.expanduser(root))
        self.train_dir = osp.join(self.dataset_dir, 'occluded_body_images')
        self.query_dir = osp.join(self.dataset_dir, 'occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'whole_body_images')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False, is_query=False)

        if verbose:
            print("=> Paritial_REID loaded")
            self.print_dataset_statistics(query, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    # ================================================================================================================
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    # ================================================================================================================
    def _process_dir(self, dir_path, relabel=False, is_query=True):
        # img_paths = glob.glob(osp.join(dir_path, '*', '*.jpg'))
        img_paths = glob.glob(osp.join(dir_path,'*.jpg'))

        if is_query:
            camid = 0
        else:
            camid = 1

        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset