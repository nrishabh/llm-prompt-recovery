#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:0
#SBATCH --job-name="ssm-project-prompt-recovery"
# save output in slurm_outputs directory
#SBATCH --output=slurm_outputs/large-generation-%j.out
#SBATCH --mem=32G

module load anaconda
conda activate ssm_project

python gen_v2.py \
    --original-datasets shakespeare-dataset email-dataset news-articles-dataset recipe-dataset resume-dataset song-lyrics-dataset \
    --prompt-dataset instruction-prompts-dataset \
    --num-prompts 10 \
    --num-originals 30 \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --quantization 4bit \
    --train-test-split 0.8 \
    --output-dataset-name "large-dataset"