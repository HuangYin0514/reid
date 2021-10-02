def pcb_script():
    from train_script.pcb_script import train

    return train


def pcb_script():
    from train_script.pcb_ft_scrip import train

    return train


def pcb_ft_ffm_script():
    from train_script.pcb_ft_ffm_script import train

    return train


def baseline_script():
    from train_script.baseline_script import train

    return train


if __name__ == "__main__":
    train = pcb_script()
    print("运行中。。。" + train.__globals__["__name__"])
    train()
