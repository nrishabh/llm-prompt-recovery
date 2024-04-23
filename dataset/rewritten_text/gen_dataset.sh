#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:0
#SBATCH --job-name="ssm-project-prompt-recovery"
#SBATCH --output=slurm-%j.out
#SBATCH --mem=32G
#SBATCH --mail-type=end 
#SBATCH --mail-user=rnanawa1@jhu.edu

module load anaconda
conda activate llm-prompt-recovery

python generation.py
