import sys
import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image


def save_fmnist_images(dataset):
    basedir = 'fmnist_images'
    count = 0
    for x, y in dataset:
        sys.stdout.write(f'{count+1} / {len(dataset)} ... \r')
        sys.stdout.flush()
        class_dir = f'{basedir}/{y}'
        os.makedirs(class_dir, exist_ok=True)
        save_image(
                x, f'{class_dir}/{count:05d}.png')
        count += 1
    return


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = datasets.FashionMNIST(
            root=sys.argv[1], train=False, download=True,
            transform=transform)
    save_fmnist_images(dataset)
