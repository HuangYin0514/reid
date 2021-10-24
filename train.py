# 参考apnet中的模块
def baseline_apnet_scrip():
    from train_script.baseline_apnet_scrip import train

    print("目标精度95.0%。。。")
    return train


def pcb_ffm_script():
    from train_script.pcb_ffm_script import train

    return train


if __name__ == "__main__":

    train = pcb_ffm_script()
    print("=" * 40)
    print("运行中1。。。" + train.__globals__["__name__"])
    print("=" * 40)
    train()
