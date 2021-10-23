# Schedule learning rate--------------------------------------------
class pcb_scheduler:
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, epoch):
        step_size = 40
        lr_new = self.lr * (0.1 ** (epoch // step_size))
        for g in self.optimizer.param_groups:
            g["lr"] = lr_new * g.get("lr_mult", 1)
