import wandb
# import json

# Initialize a new run
run = wandb.init(entity="jhu-llm-prompt-recovery", project="llm-prompt-recovery", job_type="process")

# Define the artifact name correctly without special characters
artifact_name = 'arxiv-abstracts-dataset_20240420_031345'

# Create the artifact
artifact = run.use_artifact(f'jhu-llm-prompt-recovery/llm-prompt-recovery/{artifact_name}:v0')

# Log the artifact
run.log_artifact(artifact)

# Now you can safely download it
artifact_dir = artifact.download()

# Finish the run
wandb.finish()