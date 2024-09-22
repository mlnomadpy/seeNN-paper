#!/bin/bash

# Activate your Python environment (if any)
# e.g., source /path/to/your/venv/bin/activate

# Set script arguments
MODEL_PATH="/home/bouhsi95/seenn/wandb/run-20240214_132621-ak6iqf06/files/last_exp_Edge__model.keras"
# Path to the test dataset
TEST_DATASET_PATH="/home/bouhsi95/seenn/data/experiments/normal"
dataset_path='/home/bouhsi95/seenn/data/experiments/exp1' # Path to the dataset directory
depth_path='/home/bouhsi95/seenn/data/experiments/depth' # Path to the dataset directory
normal_path='/home/bouhsi95/seenn/data/experiments/normal' # Path to the dataset directory

# Configuration parameters
NUM_CLASSES=5
IMG_HEIGHT=224
IMG_WIDTH=224
SEED=42
GPU="1"
BATCH_SIZE=256
NUM_IMG_LIM=100000

# Activate Environment
conda activate tf
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# Run test.py with the specified arguments
nohup python src/test.py --model_path $MODEL_PATH \
            --dataset_path $dataset_path \
            --depth_path $depth_path \
            --normal_path $normal_path \
            --num_classes $NUM_CLASSES \
            --img_height $IMG_HEIGHT \
            --img_width $IMG_WIDTH \
            --seed $SEED \
            --gpu $GPU \
            --batch_size $BATCH_SIZE \
            --num_img_lim $NUM_IMG_LIM > test_all.out
