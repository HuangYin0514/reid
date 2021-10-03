import os
from common import mkdirs
import time


def print_options(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        message += "{:>25}: {:<30}\n".format(str(k), str(v))
    message += "----------------- End -------------------"
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, "{}_opt.txt".format(opt.name))
    with open(file_name, "wt") as opt_file:
        opt_file.write(message)
        opt_file.write("\n")


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


def print_other_test_infomation(_, CMC, mAP, curve, logger):
    logger.info(
        "Other dataset of Testing: top1:%.4f top5:%.4f top10:%.4f mAP:%.4f"
        % (CMC[0], CMC[4], CMC[9], mAP)
    )

    # curve.x_epoch_test.append(epoch + 1)
    curve.y_other_test["top1"].append(CMC[0])
    curve.y_other_test["mAP"].append(mAP)
