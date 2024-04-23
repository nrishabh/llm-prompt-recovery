import os
import json
import torch
import wandb
import random
import pandas as pd

from datetime import datetime
from dotenv import load_dotenv
from itertools import combinations
from tqdm.autonotebook import tqdm
import transformers

load_dotenv()

JOB_TYPE = "Dataset_Generation"
CURR_DATE_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# randomly sample instructions
random.seed(42)

NUM_INSTRUCTIONS = 2
NUM_ORIGINAL_TEXTS = 100
DATASET_NAME = "arxiv-abstracts-dataset"

# TODO: No. of instructions to sample from the dataset ()

run = wandb.init(entity="jhu-llm-prompt-recovery", project="llm-prompt-recovery", job_type="upload-dataset", name=f"{JOB_TYPE}_{CURR_DATE_TIME}")

artifact = run.use_artifact("instruction-prompts-dataset:latest")
artifact_dir = artifact.download()
with open(os.path.join(artifact_dir, os.listdir(artifact_dir)[0]), "r") as f:
    prompts = json.load(f)

artifact = run.use_artifact(f"{DATASET_NAME}:latest")
artifact_dir = artifact.download()
with open(os.path.join(artifact_dir, os.listdir(artifact_dir)[0]), "r") as f:
    og_text = json.load(f)

# sample instructions
sampled_instructions = random.sample(prompts, NUM_INSTRUCTIONS)

# sample original texts
sampled_original_texts = random.sample(og_text, NUM_ORIGINAL_TEXTS)

# generate all possible combinations of instructions and original texts
# each combination will have one instruction and one original text
instruction_original_text_combinations = list(combinations([(instruction, original_text) for instruction in sampled_instructions for original_text in sampled_original_texts], 1))

rewritten_text_dataset = list()

model = "google/gemma-1.1-2b-it"

tokenizer = transformers.AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
)

# change instruction_original_text_combinations from a 2d tuple to 1d tuple
instruction_original_text_combinations = [combination[0] for combination in instruction_original_text_combinations]

for idx, (instruction, original_text) in tqdm(enumerate(instruction_original_text_combinations)):

    instruction_id = instruction['id']
    instruction_text = instruction['prompt']

    original_text_id = original_text['id']
    original_text_text = original_text['text']
    
    messages = [
        {"role": "user", "content": f"{instruction_text} {original_text_text}"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = pipeline(
        prompt,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    output = outputs[0]["generated_text"][len(prompt):]

    rewritten_text_dataset.append({
        "id": idx,
        "instruction_id": instruction_id,
        "instruction_text": instruction_text,
        "original_text_id": original_text_id,
        "original_text_text": original_text_text,
        "rewritten_text": output
    })

# write to json
with open(f"{DATASET_NAME}_{CURR_DATE_TIME}.json", "w") as f:
    json.dump(rewritten_text_dataset, f)

# upload to wandb
artifact = wandb.Artifact(f"{DATASET_NAME}_{CURR_DATE_TIME}", type="rewritten-texts-dataset")
artifact.add_file(f"{DATASET_NAME}_{CURR_DATE_TIME}.json")
run.log_artifact(artifact)

run.finish()
wandb.finish()