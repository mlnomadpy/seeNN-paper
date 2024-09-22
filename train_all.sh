#!/bin/bash

# Author: Taha Bouhsine
# Email: contact@tahabouhsine.com
# Created on: January 12th 2024
# Description: 
# This script runs the train_seenn.py script for image classification training 
# with predefined or customizable settings.

# Default parameters
learning_rate=0.001
batch_size=32
epochs=100
img_height=224
img_width=224
seed=420
gpu='1'
dataset_path='/home/bouhsi95/seenn/data/experiments/exp1' # Path to the dataset directory
depth_path='/home/bouhsi95/seenn/data/experiments/depth' # Path to the dataset directory
normal_path='/home/bouhsi95/seenn/data/experiments/normal' # Path to the dataset directory
test_dataset_path='/home/bouhsi95/seenn/data/modalities/rgb/inflight_train' # Path to the test dataset directory
num_img_lim=100000 # Number of images per class
val_split=0.2 # Validation split
n_cross_validation=5 # Number of bins for cross-validation
num_classes=5 # Number of classes
trainable_epochs=0 # Number of epochs before the backbone becomes trainable
accumulation_steps=5

# Modalities
edge=true
entropy=true
dcp=true
fft=true
spectral=true
depth=true
normal=true
rgb=true
# 

fusion_type='attention'
# Activate Environment
conda activate tf
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# Running the training script with parameters
nohup python /home/bouhsi95/seenn/src/train_all.py \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --epochs $epochs \
  --dataset_path $dataset_path \
  --depth_path $depth_path \
  --normal_path $normal_path \
  --img_height $img_height \
  --img_width $img_width \
  --seed $seed \
  --gpu $gpu \
  --test_dataset_path $test_dataset_path \
  --num_img_lim $num_img_lim \
  --val_split $val_split \
  --n_cross_validation $n_cross_validation \
  --num_classes $num_classes \
  --trainable_epochs $trainable_epochs \
  --accumulation_steps $accumulation_steps \
  --rgb $rgb \
  --normal $normal \
  --depth $depth \
  --fusion_type $fusion_type \
  > seenn_train_attention.out

  # --entropy $entropy \
  # --edge $edge \
# if you want any feature uncomment it and move it up
  # --dcp $dcp \
  # --fft $fft \
  # --spectral $spectral \
