#!/bin/bash
#SBATCH -p mit_normal_gpu            # partition (GPU queue)
#SBATCH --gres=gpu:h100:1            # request 1 GPU (adjust if needed)
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
conda activate conda_env

# go to your project directory
cd /orcd/home/002/corte911/code/finNN_code

# --- Ensure NVIDIA wheels' libs are on LD_LIBRARY_PATH ---
#export LD_LIBRARY_PATH="$(
#python - <<'PY'
#import os, glob, site
#paths=set()
#for sp in site.getsitepackages()+[site.getusersitepackages()]:
#    if not sp: 
#        continue
#    for pat in ("nvidia/*/lib","nvidia/*/lib/*"):
#        for d in glob.glob(os.path.join(sp, pat)):
#            if os.path.isdir(d):
#                paths.add(d)
#print(":".join(sorted(paths)))
#PY
#):${LD_LIBRARY_PATH}"

# === run your python script ===
python -u src/price_prediction/run_experiments.py
