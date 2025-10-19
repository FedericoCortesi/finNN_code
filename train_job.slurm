#!/bin/bash
#SBATCH -p mit_normal_gpu            # partition (GPU queue)
#SBATCH --gres=gpu:h200:1            # request 1 GPU (adjust if needed)
#SBATCH --cpus-per-task=8            # number of CPU cores
#SBATCH --mem=32G                    # memory
#SBATCH --time=06:00:00              # walltime (8 hours)
#SBATCH -J finnn_train               # job name
#SBATCH -o logs/slurm_%j.out         # stdout log
#SBATCH -e logs/slurm_%j.err         # stderr log

# ensure logs directory exists before Slurm writes to it
mkdir -p /orcd/home/002/corte911/code/finNN_code/logs

# === setup environment ===
source ~/.bashrc

# Load Anaconda Module
module load miniforge

# Activate the specific conda environment
conda activate torch_h200_py312

# go to your project directory
cd /orcd/home/002/corte911/code/finNN_code

# === run your python script ===
python -u src/run_experiments.py --config search_debug.yaml
