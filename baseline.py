import torch
import transformers
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json

# from huggingface_hub import notebook_login

# notebook_login()

# Set up logging configuration
logging.basicConfig(filename='model_outputs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data from the downloaded artifact
artifact_dir = 'arxiv-abstracts-dataset_20240420_031345'
json_name = 'arxiv-abstracts-dataset_20240420_031345.json'
with open(f'artifacts/{artifact_dir}/{json_name}', 'r') as file:
    data = json.load(file)
print('Data loaded successfully', data[0])

prompt_text = "Guess the prompt used to generate the following text:"

# Prepare input text by concatenating 'original_text_text' and 'rewritten_text'
inputs = [f"{item['original_text_text']} [SEP] {prompt_text} [SEP] {item['rewritten_text']}" for item in data]

targets = [item['instruction_text'] for item in data]

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print('Model and tokenizer loaded successfully')

# Process and generate outputs
outputs = []
for i, input_text in enumerate(inputs):
    # Encode input texts
    # encoded_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    encoded_input = tokenizer(input_text, return_tensors="pt").to(device)
    # Generate output sequences
    # output_sequences = model.generate(**encoded_input, max_length=512)
    output_sequences = model.generate(**encoded_input, max_new_tokens=50)
    # Decode the output sequences
    # output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    output_text = tokenizer.decode(output_sequences[0])
    outputs.append(output_text)
    print('Output generated successfully')
    # if i > 5:
    #     break

print('Outputs generated successfully')

# Output results, optionally compare with targets
# for output, target in zip(outputs, targets):
#     print(f"Generated: {output}\nExpected: {target}\n")

# Log results, optionally compare with targets
for output, target in zip(outputs, targets):
    logging.info(f"Generated: {output}")
    logging.info(f"Expected: {target}")

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt")

# outputs = model.generate(**input_ids)
# logging.info(tokenizer.decode(outputs[0]))





