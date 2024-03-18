#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p mlow,mlow # Partition to submit to
#SBATCH --mem 12000 # 4GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o logs/%j.out # File to which STDOUT will be written

python train_coco_metric.py