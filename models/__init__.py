
from .resnet import Res_net
from .pcb import pcb
from .pcb_rga import pcb_rga
from .pcb_cbam import resnet50_cbam_reid_model
from .pcb_cbam_v4 import resnet50_cbam_reid_model_v4
from .pcb_gloab import pcb_gloab
from .pcb_gloab_triplet import pcb_gloab_triplet
from .pcb_gloab_rga import pcb_gloab_rga
from .pcb_gloab_ffm import pcb_gloab_ffm

__model_factory = {
    'Res_net': Res_net,
    'pcb': pcb,
    'pcb_rga': pcb_rga,
    'resnet50_cbam_reid_model': resnet50_cbam_reid_model,
    'resnet50_cbam_reid_model_v4': resnet50_cbam_reid_model_v4,
    'pcb_gloab': pcb_gloab,
    'pcb_gloab_triplet': pcb_gloab_triplet,
    'pcb_gloab_rga': pcb_gloab_rga,
    'pcb_gloab_ffm':pcb_gloab_ffm

}


def build_model(num_classes, args, **kwargs):

    name = args.experiment
    height = args.img_height
    width = args.img_width

    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )

    return __model_factory[name](num_classes=num_classes, height=height, width=width)
