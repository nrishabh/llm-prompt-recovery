#!/bin/bash

#SBATCH --account=danielk_gpu
#SBATCH --partition=ica100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=cs601-llm-prompt-recovery
#SBATCH --output=outputs/slurm/large50-%J.log
#SBATCH --mail-type=all 
#SBATCH --mail-user=rnanawa1@jhu.edu

module load anaconda
conda activate rishabh

python /home/rnanawa1/llm-prompt-recovery/models/lora.py \
    --dataset nrishabh/prompt-recovery \
    --dataset_subset "large-llama-instr" \
    --model LoftQ/Meta-Llama-3-8B-Instruct-4bit-64rank \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 0.00002 \
    --epochs 50