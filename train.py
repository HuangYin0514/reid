def pcb_script():
    from train_script.baseline_script import train

    print("目标精度94.1%。。。")
    return train


# 参考apnet中的模块
def baseline_apnet_scrip():
    from train_script.baseline_apnet_scrip import train

    print("目标精度95.0%。。。")
    return train


# baseline_apnet_drop_script
def baseline_apnet_drop_script():
    from train_script.baseline_apnet_drop_script import train

    print("目标精度94.9%。。。")
    return train


def pcb_ffm_script():
    from train_script.pcb_ffm_script import train

    print("84.8|66.9 ")
    return train


if __name__ == "__main__":

    train = pcb_ffm_script()
    print("=" * 40)
    print("运行中。。。" + train.__globals__["__name__"])
    print("=" * 40)
    train()
