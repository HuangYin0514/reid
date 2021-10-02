import logging
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------------------- Logger ----------------------
class Logger(logging.Logger):
    """Inherit from logging.Logger.
    Print logs to console and file.
    """

    def __init__(self, dir_path):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        super(Logger, self).__init__("Training logger")

        # Print logs to console and file
        file_handler = logging.FileHandler(os.path.join(self.dir_path, "train_log.txt"))
        console_handler = logging.StreamHandler()
        log_format = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        self.addHandler(file_handler)
        self.addHandler(console_handler)


# Suppress pop-up windows during debugging
matplotlib.use("agg")


class Draw_Curve:
    def __init__(self, dir_path):

        self.dir_path = dir_path

        # Draw curve
        self.fig = plt.figure(clear=True)

        self.ax0 = self.fig.add_subplot(121, title="Training loss")
        self.ax1 = self.fig.add_subplot(122, title="Testing CMC/mAP")
        self.x_epoch_loss = []
        self.x_epoch_test = []

        self.y_train_loss = []
        self.y_test = {}
        self.y_test["top1"] = []
        self.y_test["mAP"] = []

        self.y_other_test = {}
        self.y_other_test["top1"] = []
        self.y_other_test["mAP"] = []

    def save_curve(self):

        # 损失函数的曲线
        self.ax0.plot(
            self.x_epoch_loss, self.y_train_loss, "bs-", markersize="2", label="test"
        )
        self._plot_min_point(self.ax0, self.y_train_loss)
        self.ax0.set_ylabel("Training")
        self.ax0.set_xlabel("Epoch")
        self.ax0.legend()

        if len(self.y_test["top1"]) > 0:
            # top1和mAP的曲线
            self.ax1.plot(
                self.x_epoch_test,
                self.y_test["top1"],
                "rs-",
                markersize="2",
                label="top1",
            )
            self._plot_max_point(self.ax1, self.y_test["top1"])
            self.ax1.plot(
                self.x_epoch_test,
                self.y_test["mAP"],
                "bs-",
                markersize="2",
                label="mAP",
            )
            self._plot_max_point(self.ax1, self.y_test["mAP"])

        if len(self.y_other_test["top1"]) > 0:
            # other dataset of top1和mAP的曲线
            self.ax1.plot(
                self.x_epoch_test,
                self.y_other_test["top1"],
                "rs-",
                markersize="2",
                label="top1",
            )
            self._plot_max_point(self.ax1, self.y_other_test["top1"])
            self.ax1.plot(
                self.x_epoch_test,
                self.y_other_test["mAP"],
                "bs-",
                markersize="2",
                label="mAP",
            )
            self._plot_max_point(self.ax1, self.y_other_test["mAP"])

            self.ax1.set_ylabel("%")
            self.ax1.set_xlabel("Epoch")
            self.ax1.legend()

        save_path = os.path.join(self.dir_path, "train_log.jpg")
        self.fig.tight_layout()  # 防止图像重叠
        self.fig.savefig(save_path)

    def _plot_min_point(self, ax, find_list):
        min_arg = np.argmin(find_list)  # 标注最小值对应的点
        min_index = min_arg + 1
        min_value = find_list[min_arg]
        ax.plot(min_index, min_value, "gs", markersize="8")
        min_showtext = "[" + str(min_index) + "," + str(round(min_value, 1)) + "]"
        ax.annotate(min_showtext, xy=(min_index, min_value))

    def _plot_max_point(self, ax, find_list):
        max_arg = np.argmax(find_list)  # 标注最小值对应的点
        max_index = max_arg + 1
        max_value = find_list[max_arg]
        ax.plot(max_index, max_value, "gs", markersize="8")
        max_showtext = "[" + str(max_index) + "," + str(round(max_value, 3)) + "]"
        ax.annotate(max_showtext, xy=(max_index, max_value))


def print_train_infomation(
    epoch, num_epochs, running_loss, train_loader, logger, curve, start_time, accuracy
):
    # print train infomation
    epoch_loss = running_loss / len(train_loader.dataset)
    time_remaining = (num_epochs - epoch) * (time.time() - start_time) / (epoch + 1)

    logger.info(
        "Epoch:{}/{} \tTrain Loss:{:.4f}\tacc:{:.2f}% \tETA:{:.0f}h{:.0f}m".format(
            epoch + 1,
            num_epochs,
            epoch_loss,
            accuracy,
            time_remaining // 3600,
            time_remaining / 60 % 60,
        )
    )
    # plot curve
    curve.x_epoch_loss.append(epoch + 1)
    curve.y_train_loss.append(epoch_loss)


def print_test_infomation(epoch, CMC, mAP, curve, logger):
    logger.info(
        "Testing: top1:%.4f top5:%.4f top10:%.4f mAP:%.4f"
        % (CMC[0], CMC[4], CMC[9], mAP)
    )

    curve.x_epoch_test.append(epoch + 1)
    curve.y_test["top1"].append(CMC[0])
    curve.y_test["mAP"].append(mAP)


def print_other_test_infomation(epoch, CMC, mAP, curve, logger):
    logger.info(
        "Other dataset of Testing: top1:%.4f top5:%.4f top10:%.4f mAP:%.4f"
        % (CMC[0], CMC[4], CMC[9], mAP)
    )

    # curve.x_epoch_test.append(epoch + 1)
    curve.y_other_test["top1"].append(CMC[0])
    curve.y_other_test["mAP"].append(mAP)
