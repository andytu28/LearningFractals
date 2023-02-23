import sys
from PIL import Image
import os
import pickle
import numpy as np
from torchvision.utils import save_image
import numpy.linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F
from fractal_datamodule.datasets.fractals import ifs


class FolderDataset(torch.utils.data.Dataset):

    def __init__(self, real_data_dir, transform=None):
        self.img_paths = []
        img_names = sorted(list(os.listdir(real_data_dir)))  # The img order is always the same
        for img_name in img_names:
            self.img_paths.append(
                    f'{real_data_dir}/{img_name}')
        self.transform = transform
        return

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(
                open(img_path, "rb")).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)


def build_global_xy_coords(size, device):
    x_coords = torch.zeros((size, size)).to(device)
    y_coords = torch.zeros((size, size)).to(device)
    for xx in range(size):
        for yy in range(size):
            x_coords[xx, yy] = xx
            y_coords[xx, yy] = yy
    return x_coords, y_coords


def normalize_coords(coords, boundary):
    xspan, yspan, xmin, ymin = boundary
    new_xs = (coords[:, 0] - xmin) / xspan
    new_ys = (coords[:, 1] - ymin) / yspan
    return torch.clamp(torch.concat([
        new_xs.unsqueeze(1),
        new_ys.unsqueeze(1)], dim=1),
        min=0, max=1)


def eval_render(coords, size, device,
                x_coords=None,
                y_coords=None, boundary=None):

    if boundary is None:
        raise ValueError()  # Assume to give boundary for eval
        boundary = setup_boundary(coords, False)

    coords = (normalize_coords(
        coords, boundary)*(size-1)).type(torch.int32)
    image = torch.zeros((size, size)).cuda()
    for c in coords:
        x, y = c
        bx = torch.clamp(x-1, min=0, max=size-1)
        ex = torch.clamp(x+1, min=0, max=size-1)
        by = torch.clamp(y-1, min=0, max=size-1)
        ey = torch.clamp(y+1, min=0, max=size-1)
        image[bx:ex, by:ey] += 1
    image = torch.clamp(image, min=0, max=1)
    return image


def setup_boundary(coords, const_boundary):
    if const_boundary:
        raise ValueError()  # we should not need this anymore
        return 10, 10, -5, -5
    coords = coords.detach().cpu().numpy()
    region = np.concatenate(
            ifs.minmax(coords))
    xspan, yspan = ifs._extent(region)
    xmin, ymin = region[0], region[1]
    return xspan, yspan, xmin, ymin


def forward_pass_render(coords, size, device,
                        x_coords=None,
                        y_coords=None, boundary=None,
                        sigma=1.):
    if boundary is None:
        boundary = setup_boundary(coords, False)

    coords = normalize_coords(coords, boundary) * (size-1)
    dist = ((coords[:, 0].unsqueeze(-1).unsqueeze(-1)-x_coords)**2+
            (coords[:, 1].unsqueeze(-1).unsqueeze(-1)-y_coords)**2)
    image = torch.sum(torch.exp(-dist/sigma), dim=0)
    return image, boundary


def build_fractal_model_svdformat(num_transforms, contractive_init=True,
                                  init_seed=831486):
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)

    ifs_w = nn.Embedding(
            num_transforms, 6)
    ifs_b = nn.Embedding(
            num_transforms, 2)
    if contractive_init:
        for index in range(num_transforms):
            theta1 = np.random.rand(1).item()*2*np.pi - np.pi
            theta2 = np.random.rand(1).item()*2*np.pi - np.pi
            sigma1 = np.random.rand(1).item()
            sigma2 = np.random.rand(1).item()
            d1 = 0.005
            d2 = 0.005
            params = torch.from_numpy(
                    np.array([theta1, theta2, sigma1, sigma2, d1, d2]).astype(
                        np.float32))
            b = torch.from_numpy((np.random.rand(2, 1) - 0.5) * 2)
            ifs_w.weight.data[index].copy_(params.view(6))
            ifs_b.weight.data[index].copy_(b.view(2))
    return ifs_w, ifs_b


def make_rotation_matrix(theta):
    r_mat = torch.concat([torch.cos(theta), -torch.sin(theta),
                          torch.sin(theta),  torch.cos(theta)]).view(2, 2)
    return r_mat


def make_diagnal_matrix(sigma_1, sigma_2):
    zero =  torch.zeros(1).to(sigma_1.device)
    d_mat = torch.concat([sigma_1,  zero,
                          zero, sigma_2]).view(2, 2)
    return d_mat


def make_matrices_from_svdformat(ifs_w_weight, force_all=True):
    all_w = []
    all_sgv = []
    for i in range(ifs_w_weight.shape[0]):
        theta_1, theta_2, sigma_1, sigma_2, d1, d2 = ifs_w_weight[i]
        r_mat1 = make_rotation_matrix(theta_1.unsqueeze(0))
        r_mat2 = make_rotation_matrix(theta_2.unsqueeze(0))
        if i == 0 or force_all:
            sig_mat = make_diagnal_matrix(
                    torch.sigmoid(sigma_1.unsqueeze(0)),
                    torch.sigmoid(sigma_2.unsqueeze(0)))
        else:
            sig_mat = make_diagnal_matrix(
                    F.softplus(sigma_1.unsqueeze(0)),
                    F.softplus(sigma_2.unsqueeze(0)))
        all_sgv.append(torch.diag(sig_mat).unsqueeze(0))
        d1 = d1.unsqueeze(0)
        d2 = d2.unsqueeze(0)
        d_mat = make_diagnal_matrix(
                d1.sign() - d1.detach() + d1,
                d2.sign() - d2.detach() + d2)
        w = torch.matmul(torch.matmul(torch.matmul(r_mat1, sig_mat), r_mat2), d_mat).T
        all_w.append(w.unsqueeze(0))
    return torch.cat(all_w, dim=0), torch.cat(all_sgv, dim=0)


def forward_pass_iterate_svdformat(ifs_w, ifs_b, seqs, start_cs):
    all_w, all_sgv = make_matrices_from_svdformat(ifs_w.weight)
    n_iters = seqs.shape[1]
    curr_cs = start_cs.view(-1, 1, 2)
    all_cs = []
    for i in range(n_iters):
        ids = seqs[:, i].long()
        w = all_w[ids]
        b = ifs_b(ids).view(-1, 1, 2)
        curr_cs = torch.bmm(curr_cs, w) + b
        all_cs.append(curr_cs)
    return torch.cat(all_cs, dim=1), all_sgv


def sampling_based_on_wacv22_svdformat(ifs_w, ifs_b, num_samples, num_coords):
    all_w, _ = make_matrices_from_svdformat(ifs_w.weight.data.detach())
    matrices = all_w
    biases = ifs_b.weight.data.detach().view(-1, 1, 2)
    determinants = (matrices[:, 0, 0]*matrices[:, 1, 1]-
                    matrices[:, 0, 1]*matrices[:, 1, 0])
    ps = torch.abs(determinants)
    ps = ps / torch.sum(ps)
    random_ints = torch.multinomial(
            ps, num_samples*num_coords, replacement=True)
    seqs = random_ints.view(num_samples, num_coords)
    s = 1 / (1+determinants[0]-matrices[0, 0, 0]-matrices[0, 1, 1])
    y = s * ((1 - matrices[0, 0, 0]) * biases[0, 0, 1] +
             matrices[0, 0, 1] * biases[0, 0, 0])
    x = s * ((1 - matrices[0, 1, 1]) * biases[0, 0, 0] +
             matrices[0, 1, 0] * biases[0, 0, 1])
    start_c = torch.cat([x[None, None], y[None, None]], dim=1)
    start_cs = start_c.repeat((num_samples, 1))
    return seqs, start_cs
