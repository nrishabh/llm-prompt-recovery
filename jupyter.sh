#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:0
#SBATCH --job-name="ssm-project-prompt-recovery"
#SBATCH --output=slurm_outputs/jupyter-%j.out
#SBATCH --mem=16G

module load anaconda
conda activate ssm_project

port=$(shuf -i8000-9999 -n1)
echo $port
node=$(hostname -s)
user=$(whoami)
jupyter-notebook --no-browser --port=${port} --ip=${node}