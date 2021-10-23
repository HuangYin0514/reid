import torch

from .pcbscheduler import pcb_scheduler


__scheduler_factory = {
    'pcb_scheduler': pcb_scheduler,
}


def build_scheduler(name,  **kwargs):
    avai_scheduler = list(__scheduler_factory.keys())
    if name not in avai_scheduler:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __scheduler_factory[name](**kwargs)
