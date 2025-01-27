#!/bin/bash

#SBATCH --account=bcaq-dtai-gh     # Your valid account
#SBATCH --job-name=simple_job      # Job name
#SBATCH --output=simple_job.%j.out # Output log file
#SBATCH --error=simple_job.%j.err  # Error log file
#SBATCH --mem=10G                  # Memory request
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=10         # CPUs per task
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --partition=ghx4           # Partition
#SBATCH --time=00:05:00            # Max runtime

source ~/.bashrc
conda activate llama_env

python ~/Research/SSGTT/a_planner_data_process.py
# python ~/Research/SSGTT/test_training.py
# python ~/Research/SSGTT/test_llama.py
