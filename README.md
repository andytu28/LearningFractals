# Learning Fractals by Gradient Descent 

This is an official pytorch implementation of "Learning Fractals by Gradient Descent," in AAAI 2023. 


## Dependencies 

* python3.6 
* torch==1.10.2
* torchvision==0.11.3
* cv2==4.5.5
* numba==0.53.1

## Usage 

We present the instructions on learning fractals for the MNIST images. Similar steps can be followed for the FMNIST and the KMNIST images. 


### Preparing the data 

Use the following command to download and store the MNIST images. 

```bash 
$ python generate_mnist.py data/
```

The MNIST dataset will be downloaded into `data/` and the images will be stored into `mnist_images/`. 


### Learning fractal parameters 

Use the following command to to reconstruct the MNIST images by learning fractal parameters. 

```bash 
$ bash run_reconstruct_100_mnist_images.sh ${GPUIDX}
```

The argument `${GPUIDX}` indicates the GPU used for training fractals, which can usually be set to 0. 

Note that this script only reconstructs 100 MNIST images (random 10 images for each of the 10 classes). The target images, the reconstructed images, and the learned fractals parameters will be stored in `IMAGEMATCH_MNIST/`.


### Evaluating the reconstruction performance 

Use the following command to compute the MSE loss between the target and the reconstructed images. 

```bash 
$ python evaluate_mse.py IMAGEMATCH_MNIST/
```

## Contact 

If you have any questions, please contact [Cheng-Hao Tu](https://andytu28.github.io/)(tu.343@osu.edu). 
