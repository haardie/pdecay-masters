#!/bin/bash

### Job Name 
#PBS -N resnet18-pl0-h5

### required runtime
#PBS -l walltime=24:00:00

### queue for submission
#PBS -q gpuA

### Merge output and error files
#PBS -j oe

#PBS -l select=1:mem=256G:ncpus=1:ngpus=4

cd /mnt/lustre/helios-home/gartmann/venv/

module load cuda/11.7
source /mnt/lustre/helios-home/gartmann/venv/bin/activate

### run the application
python single_plane_h5.py
