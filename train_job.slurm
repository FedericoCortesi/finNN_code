#!/bin/bash
#SBATCH -p mit_normal_gpu            # partition (GPU queue)
#SBATCH --gres=gpu:1                 # request 1 GPU (adjust if needed)
#SBATCH --cpus-per-task=8            # number of CPU cores
#SBATCH --mem=128G                   # memory
#SBATCH --time=06:00:00              # allocated time (6h is the max)
#SBATCH -J finnn_train               # job name
#SBATCH -o logs/slurm_%j.out         # stdout log
#SBATCH -e logs/slurm_%j.err         # stderr log

# ensure logs directory exists before Slurm writes to it
mkdir -p /orcd/home/002/corte911/code/finNN_code/logs

# === setup environment ===
source ~/.bashrc

# Load Anaconda Module

# Activate the specific conda environment
conda activate torch_h200_py312

# go to your project directory
cd /orcd/home/002/corte911/code/finNN_code

# === run your python script ===
#python -u src/run_experiments.py --config cnn_100_muon_icml_cnn_100_adam_lr.yaml
#python -u src/run_experiments.py --config cnn_100_muon_icml_cnn_100_sgd.yaml
#python -u src/run_experiments.py --config cnn_100_adam_lr_cnn_100_muon_icml.yaml
#python -u src/run_experiments.py --config cnn_100_adam_lr_cnn_100_sgd.yaml
#python -u src/run_experiments.py --config cnn_100_sgd_cnn_100_muon_icml.yaml
#python -u src/run_experiments.py --config cnn_100_sgd_cnn_100_adam_lr.yaml

#python -u src/run_experiments.py --config lstm_100_muon_lr_lstm_100_adam_lr.yaml
#python -u src/run_experiments.py --config lstm_100_muon_lr_lstm_100_sgd.yaml
#python -u src/run_experiments.py --config lstm_100_adam_lr_lstm_100_muon_lr.yaml
#python -u src/run_experiments.py --config lstm_100_adam_lr_lstm_100_sgd.yaml
#python -u src/run_experiments.py --config lstm_100_sgd_lstm_100_muon_lr.yaml
#python -u src/run_experiments.py --config lstm_100_sgd_lstm_100_adam_lr.yaml

#python -u src/run_experiments.py --config mlp_100_muon_lr_mlp_100_adam_lr.yaml
#python -u src/run_experiments.py --config mlp_100_muon_lr_mlp_100_sgd.yaml
#python -u src/run_experiments.py --config mlp_100_adam_lr_mlp_100_muon_lr.yaml
#python -u src/run_experiments.py --config mlp_100_adam_lr_mlp_100_sgd.yaml
#python -u src/run_experiments.py --config mlp_100_sgd_mlp_100_muon_lr.yaml
#python -u src/run_experiments.py --config mlp_100_sgd_mlp_100_adam_lr.yaml
