# llm-prompt-recovery
Ways to recover LLM Prompts - a NLP Self-Supervised Models project

## Directory Structure
```
llm-prompt-recovery/
│
├── dataset/                 # Contains code, logs, etc. to curate and generate datasets
│
├── models/                  # Will contain all of our models
│   ├── __init__.py
│   ├── dataset.py          # Python file for custom dataset class and related functions
│   ├── metrics.py          # Python file for all functions to calculate evaluation metrics
│   ├── SkeletonModel.py    # Base class for all models, contains functions to load dataset and evaluate results
│   ├── template.py         # Sample model which inherits the base class from SkeletonModel.py
│   ├── Baseline.py         # Kunal's model
│   ├── LoRA.py             # Rishabh's model
│   └── PrefixTuning.py     # Pristina's model
│
├── experiments/            # Directory for running experiments
│   ├── slurm_outputs/
│   ├── *.sh
│   ├── logs/
│   ├── wandb/
│   └── *.ipynb
│
└── README.md               # README file
```