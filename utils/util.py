import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import torch
from collections import OrderedDict
import matplotlib
matplotlib.use('agg')


# ---------------------- Logger ----------------------
class Logger(logging.Logger):
    '''Inherit from logging.Logger.
    Print logs to console and file.
    Add functions to draw the training log curve.'''

    def __init__(self, dir_path):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        super(Logger, self).__init__('Training logger')

        # Print logs to console and file
        file_handler = logging.FileHandler(
            os.path.join(self.dir_path, 'train_log.txt'))
        console_handler = logging.StreamHandler()
        log_format = logging.Formatter(
            "%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        self.addHandler(file_handler)
        self.addHandler(console_handler)

        # Draw curve
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121, title="Training loss")
        self.ax1 = self.fig.add_subplot(122, title="Testing CMC/mAP")
        self.x_epoch_loss = []
        self.x_epoch_test = []
        self.y_train_loss = []
        self.y_test = {}
        self.y_test['top1'] = []
        self.y_test['mAP'] = []

    def save_curve(self):
        self.ax0.plot(self.x_epoch_loss, self.y_train_loss,
                      'bs-', markersize='2', label='test')
        self.ax0.set_ylabel('Training')
        self.ax0.set_xlabel('Epoch')
        self.ax0.legend()

        self.ax1.plot(self.x_epoch_test, self.y_test['top1'],
                      'rs-', markersize='2', label='top1')
        self.ax1.plot(self.x_epoch_test, self.y_test['mAP'],
                      'bs-', markersize='2', label='mAP')
        self.ax1.set_ylabel('%')
        self.ax1.set_xlabel('Epoch')
        self.ax1.legend()

        save_path = os.path.join(self.dir_path, 'train_log.jpg')
        self.fig.savefig(save_path)

    def save_img(self, fig):
        plt.imsave(os.path.join(self.dir_path, 'rank_list.jpg'), fig)


def save_rank_list_img(query_dataloader, gallery_dataloader, sorted_index_list, sorted_true_list, junk_index_list):

    rank_lists_imgs = []

    # randomly select 10 query images to show the rank list
    query_index_list = list(range(len(sorted_index_list)))
    random.shuffle(query_index_list)
    selected_query_index = query_index_list[:10]

    for i in selected_query_index:

        cur_rank_list = []

        query_img = query_dataloader.dataset[i][0]
        query_img_with_boundary = torch.nn.functional.pad(
            query_img, (3, 3, 3, 3), "constant", value=0)
        cur_rank_list.append(query_img_with_boundary)

        sorted_index = sorted_index_list[i]
        sorted_true = sorted_true_list[i]
        junk_index = junk_index_list[i]

        # show the top 10(not junk) gallery images of the rank list
        num = 0
        idx = 0
        while num < 10:
            if sorted_index[idx] in junk_index:
                idx += 1
                continue

            gallery_img = gallery_dataloader.dataset[sorted_index[idx]][0]

            gallery_img_with_boundary = torch.nn.functional.pad(
                gallery_img, (3, 3, 3, 3), "constant", value=0)

            if sorted_true[idx]:
                # True, with green boundary
                gallery_img_with_boundary[1, :, :] = torch.nn.functional.pad(
                    gallery_img[1, :, :], (3, 3, 3, 3), "constant", value=5)
            else:
                # False, with red boundary
                gallery_img_with_boundary[0, :, :] = torch.nn.functional.pad(
                    gallery_img[0, :, :], (3, 3, 3, 3), "constant", value=5)

            cur_rank_list.append(gallery_img_with_boundary)
            idx += 1
            num += 1

        cur_rank_list_img = torch.cat(cur_rank_list, dim=2)
        cur_rank_list_img = torch.nn.functional.pad(
            cur_rank_list_img, (1, 1, 0, 0), "constant", value=0)

        rank_lists_imgs.append(cur_rank_list_img.numpy().transpose((1, 2, 0)))

    fig = np.concatenate(rank_lists_imgs, axis=0)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    fig = fig * std + mean
    fig = np.clip(fig, 0, 1)

    return fig


# ---------------------- Helper functions ----------------------
def save_network(network, path, epoch_label):
    file_path = os.path.join(path, 'net_%s.pth' % epoch_label)
    torch.save(network.state_dict(), file_path)


def load_network(network, path, epoch_label):
    # devie---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # file name -----------------------------------------------------------------------------
    file_path = os.path.join(path, 'net_%s.pth' % epoch_label)

    # Original saved file with DataParallel-------------------------------------------------------
    state_dict = torch.load(file_path, map_location=torch.device(device))

    # state dict--------------------------------------------------------------------------
    model_dict = network.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    # load model state ---->{matched_layers, discarded_layers}------------------------------------
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    network.load_state_dict(model_dict)

    # assert model state ------------------------------------------------------------------------
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

    return network

