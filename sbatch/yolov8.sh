#!/bin/bash

#SBATCH --job-name=yolov8                       # Job name
#SBATCH --output=logs/%x_%j.out                 # Standard output and error log
#SBATCH --error=logs/%x_%j.err                  # Error log
#SBATCH --partition=ls6prio                     # Partition name
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --gres=gpu:1                            # Number of GPUs
#SBATCH --exclude=gpu8a                         # only run on gpu1a,b,c

# Activate your virtual environment (if any)
source /home/ls6/hekalo/Git/xraydetection/venv/bin/activate

# Run your script
srun python src/train_yolov8.py
