import warnings
import os
import torch
from collections import OrderedDict


def save_network(network, path, epoch_label):
    file_path = os.path.join(path, "net_%s.pth" % epoch_label)
    torch.save(network.state_dict(), file_path)


def load_network(network, path, epoch_label="final"):
    # devie---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # file name -----------------------------------------------------------------------------
    file_path = os.path.join(path, "net_%s.pth" % epoch_label)

    # whether the file exists
    if not os.path.exists(file_path):
        print("file not exists")
        return

    # Original saved file with DataParallel-------------------------------------------------------
    state_dict = torch.load(file_path, map_location=torch.device(device))

    # state dict--------------------------------------------------------------------------
    model_dict = network.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    # load model state ---->{matched_layers, discarded_layers}------------------------------------
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    network.load_state_dict(model_dict)

    # assert model state ------------------------------------------------------------------------
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)".format(path)
        )
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(path))
        if len(discarded_layers) > 0:
            print(
                "** The following layers are discarded "
                "due to unmatched keys or layer size: {}".format(discarded_layers)
            )

    return network
