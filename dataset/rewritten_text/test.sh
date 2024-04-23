#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:0
#SBATCH --job-name="ssm-project-prompt-recovery"
# save output in slurm_outputs directory
#SBATCH --output=slurm_outputs/slurm-%j.out
#SBATCH --mem=16G

module load anaconda
conda activate ssm_project

python gen.py \
    --original-datasets shakespeare-dataset email-dataset news-articles-dataset recipe-dataset resume-dataset song-lyrics-dataset \
    --prompt-dataset instruction-prompts-dataset \
    --num-prompts 1 \
    --num-originals 5 \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --quantization 4bit \
    --batch-size 2 \
    --train-test-split 0.8 \
    --output-dataset-name "test-dataset"