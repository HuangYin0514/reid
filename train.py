def pcb_script():
    from train_script.baseline_script import train

    print("目标精度94.1%。。。")
    return train


# 参考apnet中的模块
def baseline_apnet_scrip():
    from train_script.baseline_apnet_scrip import train

    return train


# pcb
def pcb_script():
    from train_script.pcb_script import train

    return train


def pcb_apnet_script():
    from train_script.pcb_apnet_scrip import train

    return train


if __name__ == "__main__":

    train = baseline_apnet_scrip()
    print("=" * 40)
    print("运行中。。。" + train.__globals__["__name__"])
    print("=" * 40)
    train()
