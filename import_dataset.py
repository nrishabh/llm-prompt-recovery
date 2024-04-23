import wandb
# import json

# Initialize a new run
run = wandb.init(entity="jhu-llm-prompt-recovery", project="llm-prompt-recovery", job_type="process")

# Define the artifact name correctly without special characters
artifact_name = 'arxiv-abstracts-dataset_20240420_031345'

# Create the artifact
# artifact = run.use_artifact(artifact_name, type='rewritten-texts-dataset')
# artifact = run.use_artifact(artifact_name)
artifact = run.use_artifact(f'jhu-llm-prompt-recovery/llm-prompt-recovery/{artifact_name}:v0')

# Optionally add files or metadata to your artifact here
# artifact.add_file("your_file_here")

# Log the artifact
run.log_artifact(artifact)

# Wait for the artifact to be logged
# artifact.wait()

# Now you can safely download it
artifact_dir = artifact.download()

# Finish the run
wandb.finish()

# import wandb

# api = wandb.Api()
# artifact = api.artifact("jhu-llm-prompt-recovery/llm-prompt-recovery/rewritten-texts-dataset:latest")

# artifact.download()
