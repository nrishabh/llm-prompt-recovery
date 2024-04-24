import os
import json
import torch
import wandb
import random
import logging
import argparse
import transformers

from datetime import datetime
from dotenv import load_dotenv
from itertools import combinations
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class CombinedDataset:

    def __init__(self, args):
        self.args = args

    def fetch_original_datasets(self):

        self.original_datasets = {}

        for dataset in self.args.original_datasets:

            artifact = wandb.use_artifact(f"{dataset}:latest")
            artifact_dir = artifact.download()

            with open(
                os.path.join(artifact_dir, os.listdir(artifact_dir)[0]), "r"
            ) as f:
                self.original_datasets[dataset] = json.load(f)

            if len(self.original_datasets[dataset]) < self.args.num_originals:
                logging.warning(
                    f"Dataset {dataset} has fewer than {self.args.num_originals} items."
                )

            for item in self.original_datasets[dataset]:
                assert set(item.keys()) == {
                    "id",
                    "text",
                }, f"Dataset {dataset} is not in the correct format. Each item should have only 'id' and 'text' keys."

    def fetch_prompts(self):

        artifact = wandb.use_artifact(f"{self.args.prompt_dataset}:latest")
        artifact_dir = artifact.download()

        with open(os.path.join(artifact_dir, os.listdir(artifact_dir)[0]), "r") as f:
            self.prompt_dataset = json.load(f)

        assert (
            len(self.prompt_dataset) >= self.args.num_prompts
        ), f"Prompt dataset has fewer than {self.args.num_prompts} items."

        for item in self.prompt_dataset:
            assert set(item.keys()) == {
                "id",
                "prompt",
            }, f"Prompt dataset {self.args.prompt_dataset} is not in the correct format. Each item should have only 'id' and 'prompt' keys."

    def process_dataset(self):

        # DONE: Randomly sample instructions.
        self.sampled_instructions = random.sample(
            self.prompt_dataset, self.args.num_prompts
        )

        # DONE: Randomly sample original texts.
        self.sampled_original_texts = {
            dataset: random.sample(
                self.original_datasets[dataset], self.args.num_originals
            )
            for dataset in self.original_datasets
        }

        # DONE: Generate all possible combinations of instructions and original texts for each of the original datasets.
        self.instruction_original_text_combinations = {
            dataset: list(
                combinations(
                    [
                        (instruction, original_text)
                        for instruction in self.sampled_instructions
                        for original_text in self.sampled_original_texts[dataset]
                    ],
                    1,
                )
            )
            for dataset in self.original_datasets
        }

        # for each item in each dataset, add a field called "original_dataset"
        for dataset in self.instruction_original_text_combinations:
            for idx, item in enumerate(
                self.instruction_original_text_combinations[dataset]
            ):

                self.instruction_original_text_combinations[dataset][idx] = {
                    "instruction": item[0][0],
                    "original_text": item[0][1],
                }
                self.instruction_original_text_combinations[dataset][idx][
                    "original_text"
                ]["dataset"] = dataset

    def train_test_split(self):
        self.train_datasets = {}
        self.test_datasets = {}
        for dataset in self.instruction_original_text_combinations:
            # Shuffle the combinations
            random.shuffle(self.instruction_original_text_combinations[dataset])
            # Split into train and test
            split_index = int(
                self.args.train_test_split
                * len(self.instruction_original_text_combinations[dataset])
            )
            self.train_datasets[dataset] = self.instruction_original_text_combinations[
                dataset
            ][:split_index]
            self.test_datasets[dataset] = self.instruction_original_text_combinations[
                dataset
            ][split_index:]

    def flatten(self):
        self.train_dataset = []
        self.test_dataset = []
        for dataset in self.train_datasets:
            self.train_dataset.extend(self.train_datasets[dataset])
            self.test_dataset.extend(self.test_datasets[dataset])

        if self.args.max_items is not None:
            self.train_dataset = random.sample(
                self.train_dataset, self.args.max_items * self.args.train_test_split
            )
            self.test_dataset = random.sample(
                self.test_dataset,
                self.args.max_items
                - (self.args.max_items * self.args.train_test_split),
            )


class Dataloader:

    def __init__(self, dataset, tokenizer):

        self.tokenizer = tokenizer
        self.detailed_dataset = dataset
        self.dataset = list()

    def process(self):

        for idx, item in tqdm(
            enumerate(self.detailed_dataset),
            desc="Tokenizing data items",
            unit="data item",
            total=len(self.detailed_dataset),
        ):
            messages = [
                {"role": "system", "content": item["instruction"]["prompt"]},
                {"role": "user", "content": item["original_text"]["text"]},
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )

            self.dataset.append(input_ids)

    def __len__(self):
        return len(self.detailed_dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def main(args):
    JOB_TYPE = "generate-rewritten-texts"
    CURR_DATE_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

    # DONE: Initialize wandb run
    wandb.init(
        entity="jhu-llm-prompt-recovery",
        project="llm-prompt-recovery",
        job_type=JOB_TYPE,
        name=f"{JOB_TYPE}_{CURR_DATE_TIME}",
        config=args,
    )

    dataset = CombinedDataset(args)
    dataset.fetch_original_datasets()
    dataset.fetch_prompts()
    dataset.process_dataset()
    dataset.train_test_split()
    dataset.flatten()

    if args.quantization == "8bit":
        model_kwargs = {
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_8bit": True},
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
    elif args.quantization == "4bit":
        model_kwargs = {
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": True},
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
    else:
        model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )

    logging.info("Processing train dataset...")
    train_dataloader = Dataloader(dataset.train_dataset, tokenizer=tokenizer)
    train_dataloader.process()

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    for idx, inputs in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc="Generating rewritten text",
        unit="data item",
    ):
        outputs = model.generate(
            inputs.to(model.device),
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = outputs[0][inputs.shape[-1] :]
        dataset.train_dataset[idx]["rewritten_text"] = tokenizer.decode(
            response, skip_special_tokens=True
        )

    # DONE: Save the datasets
    if args.output_dataset_name is None:
        output_dataset_name = f"{JOB_TYPE}_{CURR_DATE_TIME}"
    else:
        output_dataset_name = args.output_dataset_name

    if not os.path.exists("artifacts/" + output_dataset_name):
        os.makedirs("artifacts/" + output_dataset_name)
    with open(f"artifacts/{output_dataset_name}/train.json", "w") as f:
        json.dump(dataset.train_dataset, f)
    with open(f"artifacts/{output_dataset_name}/test.json", "w") as f:
        json.dump(dataset.test_dataset, f)

    # DONE: Upload to wandb
    artifact = wandb.Artifact(output_dataset_name, type="rewritten-texts-dataset")
    artifact.add_file(f"artifacts/{output_dataset_name}/train.json")
    artifact.add_file(f"artifacts/{output_dataset_name}/test.json")
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument(
        "--original-datasets",
        nargs="+",
        help="List of datasets to fetch and process from wandb",
    )
    parser.add_argument(
        "--prompt-dataset",
        default="instruction-prompts-dataset",
        help="Name of prompt dataset to fetch from wandb",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts to consider"
    )
    parser.add_argument(
        "--num-originals",
        type=int,
        default=1000,
        help="Number of texts to consider from each original dataset file",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Open-source llm model to use (fetched from HuggingFace)",
    )
    # modify the parser argument called "quantization" which has options as 8bit or 4bit. User can choose to leave it blank too, which will default to "no quatinzation".
    parser.add_argument(
        "--quantization",
        choices=["8bit", "4bit", None],
        default=None,
        help="Quantization of the model (options: 8bit, 4bit, or leave blank for no quantization)",
    )
    parser.add_argument(
        "--train-test-split", type=float, default=0.8, help="Train test split ratio"
    )
    parser.add_argument(
        "--output-dataset-name",
        default=None,
        help="Output dataset name (default: current date and time)",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items the dataset should have (train+test total)",
    )

    load_dotenv()

    main(parser.parse_args())
