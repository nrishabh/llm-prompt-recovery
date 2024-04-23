#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:0
#SBATCH --job-name="ssm-project-prompt-recovery"
# save output in slurm_outputs directory
#SBATCH --output=slurm_outputs/slurm-%j.out
#SBATCH --mem=32G
#SBATCH --mail-type=end 
#SBATCH --mail-user=rnanawa1@jhu.edu

module load anaconda
conda activate llm-prompt-recovery

# parser.add_argument(
#         "--original-datasets",
#         nargs="+",
#         help="List of datasets to fetch and process from wandb",
#     )
#     parser.add_argument(
#         "--prompt-dataset",
#         default="instruction-prompts-dataset",
#         help="Name of prompt dataset to fetch from wandb",
#     )
#     parser.add_argument(
#         "--num-prompts", type=int, default=10, help="Number of prompts to consider"
#     )
#     parser.add_argument(
#         "--num-originals",
#         type=int,
#         default=1000,
#         help="Number of texts to consider from each original dataset file",
#     )
#     parser.add_argument(
#         "--model",
#         default="meta-llama/Meta-Llama-3-8B-Instruct",
#         help="Open-source llm model to use (fetched from HuggingFace)",
#     )
#     parser.add_argument(
#         "--quantization",
#         choices=["8bit", "4bit"],
#         default="8bit",
#         help="Quantization options",
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         help="Number of prompts to process in a single inference",
#     )
#     parser.add_argument(
#         "--train-test-split", type=float, default=0.8, help="Train test split ratio"
#     )
#     parser.add_argument(
#         "--output-dataset-name",
#         default=None,
#         help="Output dataset name (default: current date and time)",
#     )

# names of original text datasets
# yahoo-dataset
# shakespeare-dataset
# email-dataset
# news-articles-dataset
# recipe-dataset
# resume-dataset
# song-lyrics-dataset
# arxiv-abstracts-dataset

python generation.py \
    --original-datasets yahoo-dataset shakespeare-dataset email-dataset news-articles-dataset recipe-dataset resume-dataset song-lyrics-dataset arxiv-abstracts-dataset \
    --prompt-dataset instruction-prompts-dataset \
    --num-prompts 10 \
    --num-originals 100 \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --quantization 8bit \
    --batch-size 10 \
    --train-test-split 0.8 \
    --output-dataset-name "prompt-recovery-dataset"