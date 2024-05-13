#!/bin/bash

#SBATCH --account=danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --job-name=cs601-llm-prompt-recovery
#SBATCH --output=outputs/slurm/large-%J.log
#SBATCH --mail-type=all 
#SBATCH --mail-user=rnanawa1@jhu.edu

module load anaconda
conda activate rishabh

python /home/rnanawa1/llm-prompt-recovery/dataset/rewritten_text/gen_v2.py \
    --original-datasets shakespeare-dataset email-dataset news-articles-dataset recipe-dataset resume-dataset song-lyrics-dataset \
    --prompt-dataset instruction-prompts-dataset \
    --num-prompts 10 \
    --num-originals 30 \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --quantization 4bit \
    --train-test-split 0.8 \
    --output-dataset-name "large-dataset"