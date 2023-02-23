import sys
import os
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision.utils import save_image
from torchvision import transforms
import cv2
import numpy as np
import numpy.linalg as LA
import pickle
import argparse
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


def set_logger(path, concat_logs=False):
    global logger
    m = 'a' if concat_logs else 'w'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path, mode=m)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def run_train(args, output_dir, opti_ifs_w, opti_ifs_b, tar_images, device):
    num_iters = 1000 * args.num_iters_times
    save_divi = 2

    torch.save({'w': opti_ifs_w.weight.data,
                'b': opti_ifs_b.weight.data},
                f'{output_dir}/iter0_opti_ifs_code.pth')

    optimizer = optim.Adam(
            [opti_ifs_w.weight, opti_ifs_b.weight], lr=args.lr)
    seqs, start_cs = None, None
    boundary = (10, 10, -5, -5)  # default boundary

    # Main training loop
    for iteration in range(1, num_iters+1):
        optimizer.zero_grad()

        # Add random noise
        if args.noise > 0 and iteration % 10 == 0 and iteration < 600:
            opti_ifs_w.weight.data = opti_ifs_w.weight.data + torch.randn_like(
                    opti_ifs_w.weight.data)*args.noise

        # Foward pass to render images from the ifs code model
        seqs, start_cs = sampling_based_on_wacv22_svdformat(
                opti_ifs_w, opti_ifs_b, args.gen_batch_size, args.num_coords)
        opti_coords, _ = forward_pass_iterate_svdformat(
                opti_ifs_w, opti_ifs_b, seqs, start_cs)

        # Generate for each sample in the batch
        gen_images = []
        for index in range(args.gen_batch_size):
            image, boundary = forward_pass_render(
                    opti_coords[index], args.image_size, device,
                    x_coords=x_coords, y_coords=y_coords, boundary=boundary,
                    sigma=args.std)
            gen_images.append(image[None, None, ...])

        gen_images = torch.clamp(torch.cat(gen_images, dim=0), min=0, max=1)

        # Compute the loss
        loss = F.mse_loss(gen_images.type(torch.float32),
                          tar_images.type(torch.float32),
                          reduction='mean')

        # Optimization step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opti_ifs_w.weight, 1.0)
        torch.nn.utils.clip_grad_norm_(opti_ifs_b.weight, 1.0)
        optimizer.step()

        # Learning rates decay
        if iteration % int(num_iters//4) == 0:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5

        # Save the ifs code model
        if iteration % (int(num_iters//save_divi)) == 0 or iteration == num_iters:
            torch.save({'w': opti_ifs_w.weight.data,
                        'b': opti_ifs_b.weight.data},
                        f'{output_dir}/iter{iteration}_opti_ifs_code.pth')

        logging.info(f'{iteration}: {loss.item():.6f}')

    # Render the optimized images for different iterations
    num_show_images = 1
    for iteration in range(0, num_iters+1):
        if (num_iters != 0 and (iteration % (int(num_iters//save_divi)) == 0)
            or iteration == num_iters
            or iteration == 0):
            content = torch.load(f'{output_dir}/iter{iteration}_opti_ifs_code.pth')
            opti_ifs_w.weight.data.copy_(content['w'])
            opti_ifs_b.weight.data.copy_(content['b'])

            with torch.no_grad():
                seqs, start_cs = sampling_based_on_wacv22_svdformat(
                        opti_ifs_w, opti_ifs_b, num_show_images,
                        args.num_coords*args.show_num_coords_times)
                opti_coords, _ = forward_pass_iterate_svdformat(
                        opti_ifs_w, opti_ifs_b, seqs, start_cs)

                opti_images = []
                eval_images = []
                for index in range(num_show_images):

                    oimage, _ = forward_pass_render(
                            opti_coords[index],
                            args.image_size, device,
                            x_coords=x_coords,
                            y_coords=y_coords,
                            boundary=boundary,
                            sigma=args.std)

                    eimage = eval_render(
                            opti_coords[index],
                            args.image_size, device,
                            x_coords=x_coords,
                            y_coords=y_coords,
                            boundary=boundary)

                    opti_images.append(oimage[None, None, ...])
                    eval_images.append(eimage[None, None, ...])

                opti_images = torch.clamp(
                        torch.cat(opti_images, dim=0), min=0, max=1)
                eval_images = torch.clamp(
                        torch.cat(eval_images, dim=0), min=0, max=1)

                save_image(opti_images.data[:num_show_images].cpu(),
                        f'{output_dir}/iter{iteration}_opti_images.png', nrow=1)
                save_image(eval_images.data[:num_show_images].cpu(),
                        f'{output_dir}/iter{iteration}_eval_images.png', nrow=1)
    return


def get_target_images(args, device, x_coords, y_coords):
    if args.target == 'mnist':
        data_dir = f'mnist_images/{args.idx}'
    elif args.target == 'fmnist':
        data_dir = f'fmnist_images/{args.idx}'
    elif args.target == 'kmnist':
        data_dir = f'kmnist_images/{args.idx}'
    else:
        raise NotImplementedError()

    if args.target == 'fmnist' or args.target == 'kmnist':
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.2).type(torch.float32))  # binarize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])

    if args.sample_idx != -1:
        assert(args.tar_batch_size == 1)
        dataset = Subset(FolderDataset(data_dir, transform),
                [args.sample_idx])
    else:
        raise NotImplementedError()

    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.tar_batch_size, shuffle=False)
    tar_images = next(iter(dataloader)).to(device)

    if args.tar_batch_size == 1:  # one code one image
        repeat_num = args.gen_batch_size
    else:
        raise NotImplementedError()

    logging.info(f'tar_images original sahpe: {tar_images.shape}')
    tar_images = tar_images.repeat((repeat_num, 1, 1, 1))
    return tar_images


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Experiement Settings
    parser.add_argument('--init_seed', type=int, default=831486,
                        help='Random seed for initialization.')
    parser.add_argument('--target', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'kmnist'],
                        help='Target image type for training ifs codes.')
    parser.add_argument('--idx', type=int, default=0,
                        help='Index of the downstream data.')
    parser.add_argument('--sample_idx', type=int, default=-1,
                        help='Index of the target sample when tar_batch_size=1.')

    # Training Settings
    parser.add_argument('--num_coords', type=int, default=300,
                        help='Number of coordiates.')
    parser.add_argument('--show_num_coords_times', type=int, default=2,
                        help='Times of number of coordiates for showing.')
    parser.add_argument('--num_iters_times', type=int, default=1,
                        help='Times of number of iterations.')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Image size for rendering.')
    parser.add_argument('--num_transforms', type=int, default=2,
                        help='Number of transformations in ifs.')

    parser.add_argument('--tar_batch_size', type=int, default=50,
                        help='Number of target images in a batch.')
    parser.add_argument('--gen_batch_size', type=int, default=50,
                        help='Number of generated images in a batch.')

    # Optimization Settings
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--std', type=float, default=1,
                        help='Std for rendering')
    parser.add_argument('--noise', type=float, default=0,
                        help='Noise for SGD.')

    # Other Options
    parser.add_argument('--postfix', type=str, default='',
                        help='Postfix for the inner_dir.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    torch.random.manual_seed(0xCAFFE)
    np.random.seed(0xCAFFE)
    device = torch.device('cuda')

    inner_dir  = f'img{args.image_size}'
    inner_dir += f'_nc{args.num_coords}'
    inner_dir += f'_tbs{args.tar_batch_size}'
    inner_dir += f'_gbs{args.gen_batch_size}'
    inner_dir += f'_nt{args.num_transforms}'
    inner_dir += f'_lr{args.lr}'
    inner_dir += f'_std{args.std}'
    inner_dir += f'_n{args.noise}'
    inner_dir += f'_initseed{args.init_seed}'
    inner_dir += f'{args.postfix}'
    output_dir = f'IMAGEMATCH_{args.target.upper()}/{inner_dir}/IDX{args.idx}'

    if args.sample_idx != -1:
        output_dir += f'-Sample{args.sample_idx}'
        assert(args.tar_batch_size == 1)
    else:
        raise NotImplementedError()

    os.makedirs(output_dir, exist_ok=True)
    set_logger(os.path.join(output_dir, 'log.txt'))

    x_coords, y_coords = build_global_xy_coords(
            args.image_size, device)

    logging.info(f'Learning {args.target.upper()} IDX{args.idx} ...')
    tar_images = get_target_images(args, device,
            x_coords=x_coords, y_coords=y_coords)
    save_image(tar_images.data[:1].cpu(),
            f'{output_dir}/tar_images.png', nrow=1)

    opti_ifs_w, opti_ifs_b = build_fractal_model_svdformat(
            num_transforms=args.num_transforms, contractive_init=True,
            init_seed=args.init_seed)
    opti_ifs_w.to(device)
    opti_ifs_b.to(device)

    logging.info(f'Run the training loop ...')
    run_train(args, output_dir, opti_ifs_w, opti_ifs_b,
              tar_images, device)
