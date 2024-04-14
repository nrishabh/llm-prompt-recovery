import wandb

def upload_to_wandb(local_path, artifact_name, artifact_type):
    run = wandb.init(entity="jhu-llm-prompt-recovery", project="llm-prompt-recovery", job_type="upload-dataset")
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(local_path=local_path)
    run.log_artifact(artifact)
    wandb.finish()
wandb.init()

if __name__ == "__main__":

    # local_path = "arxiv_abstracts.json"
    # artifact_name = "arxiv-abstracts-dataset"
    # local_path = "song_lyrics.json"
    # artifact_name = "song-lyrics-dataset"
    # local_path = "resume.json"
    # artifact_name = "resume-dataset"
    # local_path = "recipes.json"
    # artifact_name = "recipe-dataset"
    local_path = "prompts100.json"
    artifact_name = "prompts-dataset"

    upload_to_wandb(local_path, artifact_name, "dataset")