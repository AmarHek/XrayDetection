#!/bin/bash

#SBATCH --job-name=detectron2                   # Job name
#SBATCH --output=logs/%x_%j.out                 # Standard output and error log
#SBATCH --error=logs/%x_%j.err                  # Error log
#SBATCH --partition=ls6prio                     # Partition name
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --gres=gpu:1                            # Number of GPUs
#SBATCH --exclude=gpu8a                         # only run on gpu1a,b,c,d

# Activate your virtual environment (if any)
source /home/ls6/hekalo/Git/xraydetection/venv/bin/activate
export PATH=/usr/local/cuda-11.8/bin/${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.8

# Run your script
srun python src/detectron2/detectron2.py
