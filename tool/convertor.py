# convertor: extract weights from torch model, then save them into
# a npz file.
# original vgg_normalised torch model (.t7 format) download link:
# https://s3.amazonaws.com/xunhuang-public/adain/vgg_normalised.t7
# torch model credits: Xun Huang (https://github.com/xunhuang1995)

import torch
import numpy as np
from torch.utils.serialization import load_lua


TORCH_MODEL_PATH = 'vgg_normalised.t7'
NPZ_OUTPUT_PATH  = 'vgg19_normalised.npz'
WEIGHTS_INDICES  = (2, 5, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 42, 45, 48, 51)


def convert(src_model_path, dst_model_path, weights_indices):
    model = load_lua(src_model_path)

    weights = []
    for idx in weights_indices:
        kernel = model.modules[idx].weight.numpy()
        bias   = model.modules[idx].bias.numpy()

        weights.append(kernel)
        weights.append(bias)

    np.savez(dst_model_path, *weights)


if __name__ == '__main__':
    convert(TORCH_MODEL_PATH, NPZ_OUTPUT_PATH, WEIGHTS_INDICES)

