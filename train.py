

def pcb_script():
    from train_script.baseline import train

    return train


if __name__ == "__main__":

    train = pcb_script()
    print("运行中。。。" + train.__globals__["__name__"])
    train()
