#!/bin/bash

#SBATCH --account=danielk_gpu
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=cs601-llm-prompt-recovery
#SBATCH --output=outputs/jupyter/jupyter-%J.log
#SBATCH --mail-type=all 
#SBATCH --mail-user=rnanawa1@jhu.edu

module load anaconda
conda activate rishabh

port=$(shuf -i8000-9999 -n1)
echo $port
node=$(hostname -s)
user=$(whoami)
jupyter-notebook --no-browser --port=${port} --ip=${node}