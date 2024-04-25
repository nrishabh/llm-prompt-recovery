import torch
import transformers
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

from dotenv import load_dotenv

load_dotenv()

# Set up logging configuration
logging.basicConfig(filename='baseline_it.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize a new run
run = wandb.init(entity="jhu-llm-prompt-recovery", project="llm-prompt-recovery", job_type="process")

# Define the artifact name correctly without special characters
artifact_name = 'mini-dataset'

# Create the artifact
artifact = run.use_artifact(f'jhu-llm-prompt-recovery/llm-prompt-recovery/{artifact_name}:latest')

# Now you can safely download it
artifact_dir = artifact.download()

run.finish()

with open(f'{artifact_dir}/train.json', 'r') as file:
    data = json.load(file)
print('Data loaded successfully', data[0])

# Prepare input text by concatenating 'original_text_text' and 'rewritten_text'
# inputs = [f"{item['original_text_text']} [SEP] {prompt_text} [SEP] {item['rewritten_text']}" for item in data]
targets = [item['instruction']['prompt'] for item in data]
original_text_inputs = [item['original_text']['text'] for item in data]
rewritten_text_inputs = [item['rewritten_text'] for item in data]

# Load tokenizer and model
model_id = "gg-hf/gemma-2b-it"
dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype=dtype,
)
print('Model and tokenizer loaded successfully')

# Prepare chat template
prompt_text = "Generate the prompt used to rewrite the above text into the following text"

# chat = [
#     { "role": "user", "content": "Write a hello world program" },
# ]
# prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Process and generate outputs
outputs = []
bleu_scores = []
for original_text, rewritten_text, target in tqdm(zip(original_text_inputs, rewritten_text_inputs, targets), total=len(original_text_inputs)):
    # Creating chat for gemma 2b it
    # chat = [
    #     { "role": "user", "content": original_text },
    #     { "role": "user", "content": prompt_text },
    #     { "role": "user", "content": rewritten_text },
    # ]

    chat = [
        { "role": "user", "content": f'{original_text} [SEP] {prompt_text} [SEP] {rewritten_text}' },
    ]

    # tokenizing the prompt for the model
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Encode input texts
    encoded_input = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    # Generate output sequences
    output_sequences = model.generate(input_ids=encoded_input.to(model.device), max_new_tokens=150)
    # Decode the output sequences
    output_text = tokenizer.decode(output_sequences[0])
    outputs.append(output_text)
    
    # Compute BLEU score
    reference = [target.split()]
    candidate = output_text.split()
    score = sentence_bleu(reference, candidate)
    bleu_scores.append(score)

# Log results, and compute BLUE Score
for output, target, bleu in zip(outputs, targets, bleu_scores):
    logging.info(f"Generated: {output}")
    logging.info(f"Expected: {target}")
    logging.info(f"BLEU Score: {bleu}")

average_bleu = sum(bleu_scores) / len(bleu_scores)
logging.info(f'Average BLEU score: {average_bleu}')