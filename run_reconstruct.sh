#!/bin/bash

GPU=$1
TARGET=$2
IDX=$3
SAMPLE_IDX=$4
LR=0.05
INIT_SEED=100


CUDA_VISIBLE_DEVICES=$GPU python train_deep_fractal.py \
  --target $TARGET --idx $IDX --sample_idx $SAMPLE_IDX \
  --num_coords 300 --image_size 32 --num_transforms 10 --tar_batch_size 1 \
  --gen_batch_size 50 --lr $LR --std 1 --noise 0.1 --init_seed $INIT_SEED
