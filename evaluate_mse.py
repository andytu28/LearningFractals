from types import SimpleNamespace
import copy
import random
import numpy as np
import sys
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Subset
from torchvision.utils import save_image, make_grid
from train_deep_fractal import get_target_images
from deep_fractal import (
        forward_pass_iterate_svdformat,
        sampling_based_on_wacv22_svdformat,
        build_fractal_model_svdformat
)
from deep_fractal import (
        forward_pass_render,
        build_global_xy_coords,
        eval_render
)
from deep_fractal import FolderDataset


BASEDIR = sys.argv[1]
if BASEDIR[-1] == '/': BASEDIR = BASEDIR[:-1]

LR = 0.05
INIT_SEED = 100

settings = [
        'groundtruth',
        f'img32_nc300_tbs1_gbs50_nt10_lr{LR}_std1.0_n0.1_initseed{INIT_SEED}'
]

tokens = BASEDIR.split('/')
target = tokens[0][len('IMAGEMATCH_'):]
assert(len(tokens) == 1)
num_rand_seq_for_eval = 100
all_inner_dirs = list(os.listdir(f'{BASEDIR}/{settings[1]}'))
all_inner_dirs = sorted(
        all_inner_dirs, key=lambda x: int(x.split('-')[0][3:]))

full_mses = []
for inner_index, inner_dir in enumerate(all_inner_dirs):

    all_mses = []
    inner_tokens = inner_dir.split('-')
    IDX = int(inner_tokens[0][3:])
    SAMPLE = int(inner_tokens[1][6:])

    print(f'{inner_index}/{len(all_inner_dirs)}')
    print(f'{inner_dir} -> {IDX}:{SAMPLE}')

    for setting_idx, setting in enumerate(settings):
        print(setting)

        if setting == 'groundtruth':
            args = SimpleNamespace()
            args.target = target.lower()
            args.idx = IDX
            args.tar_batch_size = 1
            args.gen_batch_size = 1
            args.image_size = 32
            args.sample_idx = SAMPLE

            device = torch.device('cuda')
            x_coords, y_coords = build_global_xy_coords(
                    args.image_size, device)
            tar_image = get_target_images(
                    args, device, x_coords, y_coords)
            max_val = torch.max(tar_image).item()
            min_val = torch.min(tar_image).item()
            assert(max_val <= 1 and max_val >= 0)
            assert(min_val <= 1 and min_val >= 0)
            all_mses.append(0.0)

        else:
            iteration = 1000
            dir_path = f'{BASEDIR}/{setting}/IDX{IDX}-Sample{SAMPLE}'
            model_name = f'iter{iteration}_opti_ifs_code.pth'
            model_path = os.path.join(dir_path, model_name)

            if not os.path.exists(model_path):
                raise NotImplementedError()

            num_transforms = 10
            size = 32
            num_coords = 300

            # Prepare the fractal model
            device = torch.device('cuda')
            x_coords, y_coords = build_global_xy_coords(
                    size, device)
            opti_ifs_w, opti_ifs_b = build_fractal_model_svdformat(
                    num_transforms, contractive_init=True)
            content = torch.load(model_path)
            opti_ifs_w.weight.data.copy_(content['w'])
            opti_ifs_b.weight.data.copy_(content['b'])
            opti_ifs_w.to(device)
            opti_ifs_b.to(device)

            # Compute the smallest mse by sampling from 100 rand sequences
            boundary = (10, 10, -5, -5)  # default boundary
            with torch.no_grad():

                # ===== Sampling transformations =====
                seqs, start_cs = sampling_based_on_wacv22_svdformat(
                        opti_ifs_w, opti_ifs_b, num_rand_seq_for_eval, num_coords)

                # ===== Generating coordinates =====
                opti_coords, _ = forward_pass_iterate_svdformat(
                        opti_ifs_w, opti_ifs_b, seqs, start_cs)

                gen_images = []
                for i in range(num_rand_seq_for_eval):
                    image, _ = forward_pass_render(
                            opti_coords[i], size, device, x_coords=x_coords,
                            y_coords=y_coords, boundary=boundary,
                            sigma=2. if inner_index == 6 else 1.)
                    gen_images.append(image[None, None, ...])
                gen_images = torch.clamp(torch.cat(gen_images, dim=0), min=0, max=1)
                all_mses.append(torch.min(
                        torch.mean((gen_images-tar_image)**2, dim=(1, 2, 3))
                        ).detach().cpu().numpy())

    assert(len(all_mses) == 2)
    full_mses.append(all_mses)

# Compute the overall mse results
full_mses = np.array(full_mses)
print(full_mses.shape)
print(np.mean(full_mses, axis=0))
