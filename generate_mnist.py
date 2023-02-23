import sys
import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image


def save_mnist_images(dataset):
    basedir = 'mnist_images'
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
    dataset = datasets.MNIST(
            sys.argv[1],
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(32),
                 transforms.ToTensor()]))
    save_mnist_images(dataset)
