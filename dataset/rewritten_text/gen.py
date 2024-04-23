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


class LangModel:

    def __init__(self, args):

        if args.quantization == "8bit":
            model_kwargs = {
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_8bit": True},
                "low_cpu_mem_usage": True,
            }
            device = None
        elif args.quantization == "4bit":
            model_kwargs = {
                "torch_dtype": torch.float32,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            }
            device = None
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_kwargs = {"torch_dtype": torch.float16}

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=args.model,
            model_kwargs=model_kwargs,
            device=device,
        )

        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def generate_text(self, dataset):

        messages = list()
        for item in dataset:
            messages.append(
                [
                    {"role": "system", "content": item["instruction"]["prompt"]},
                    {"role": "user", "content": item["original_text"]["text"]},
                ]
            )

        prompts = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipeline(
            prompts,
            max_new_tokens=1000,
            eos_token_id=self.terminators,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        for idx, output in enumerate(outputs):
            dataset[idx]["rewritten_text"] = output[0]["generated_text"][
                len(prompts[idx]) :
            ]

        return dataset


def main(args):

    JOB_TYPE = "generate-rewritten-texts"
    CURR_DATE_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

    # DONE: Initialize wandb run
    wandb.init(
        entity="jhu-llm-prompt-recovery",
        project="llm-prompt-recovery",
        job_type=JOB_TYPE,
        name=f"{JOB_TYPE}_{CURR_DATE_TIME}",
    )

    # DONE: Fetch orignal datasets. Raise error if not found.
    original_datasets = {}
    for dataset in args.original_datasets:
        artifact = wandb.use_artifact(f"{dataset}:latest")
        artifact_dir = artifact.download()
        with open(os.path.join(artifact_dir, os.listdir(artifact_dir)[0]), "r") as f:
            original_datasets[dataset] = json.load(f)
        # check that the length of dataset is at least args.num_originals
        if len(original_datasets[dataset]) < args.num_originals:
            logging.warning(
                f"Dataset {dataset} has fewer than {args.num_originals} items."
            )
        for item in original_datasets[dataset]:
            assert set(item.keys()) == {
                "id",
                "text",
            }, f"Dataset {dataset} is not in the correct format. Each item should have only 'id' and 'text' keys."

    # DONE: Fetch prompt dataset. Raise error if not found.
    artifact = wandb.use_artifact(f"{args.prompt_dataset}:latest")
    artifact_dir = artifact.download()
    with open(os.path.join(artifact_dir, os.listdir(artifact_dir)[0]), "r") as f:
        prompt_dataset = json.load(f)
    # if no. of items in prompt dataset are less than args.num_prompts, raise error
    assert (
        len(prompt_dataset) >= args.num_prompts
    ), f"Prompt dataset has fewer than {args.num_prompts} items."
    for item in prompt_dataset:
        assert set(item.keys()) == {
            "id",
            "prompt",
        }, f"Prompt dataset {args.prompt_dataset} is not in the correct format. Each item should have only 'id' and 'prompt' keys."

    # DONE: Randomly sample instructions.
    sampled_instructions = random.sample(prompt_dataset, args.num_prompts)

    # DONE: Randomly sample original texts.
    sampled_original_texts = {
        dataset: random.sample(original_datasets[dataset], args.num_originals)
        for dataset in original_datasets
    }

    # DONE: Generate all possible combinations of instructions and original texts for each of the original datasets.
    instruction_original_text_combinations = {
        dataset: list(
            combinations(
                [
                    (instruction, original_text)
                    for instruction in sampled_instructions
                    for original_text in sampled_original_texts[dataset]
                ],
                1,
            )
        )
        for dataset in original_datasets
    }

    # for each item in each dataset, add a field called "original_dataset"
    for dataset in instruction_original_text_combinations:
        for idx, item in enumerate(instruction_original_text_combinations[dataset]):
            # print(f"Item: {item}")
            instruction_original_text_combinations[dataset][idx] = {
                "instruction": item[0][0],
                "original_text": item[0][1],
            }
            instruction_original_text_combinations[dataset][idx]["original_text"][
                "dataset"
            ] = dataset

    # DONE: Split the dataset into train and test for each dataset.
    train_datasets = {}
    test_datasets = {}
    for dataset in instruction_original_text_combinations:
        # Shuffle the combinations
        random.shuffle(instruction_original_text_combinations[dataset])
        # Split into train and test
        split_index = int(
            args.train_test_split * len(instruction_original_text_combinations[dataset])
        )
        train_datasets[dataset] = instruction_original_text_combinations[dataset][
            :split_index
        ]
        test_datasets[dataset] = instruction_original_text_combinations[dataset][
            split_index:
        ]

    # DONE: Flatten the dataset
    train_dataset = []
    test_dataset = []
    for dataset in train_datasets:
        train_dataset.extend(train_datasets[dataset])
        test_dataset.extend(test_datasets[dataset])

    # DONE: Batch both the datasets
    # If the batch size is not perfectly divisble by the number of items in the dataset, the last batch will have fewer items.
    if args.batch_size is not None:
        train_dataset_batches = [
            train_dataset[i : i + args.batch_size]
            for i in range(0, len(train_dataset), args.batch_size)
        ]
        test_dataset_batches = [
            test_dataset[i : i + args.batch_size]
            for i in range(0, len(test_dataset), args.batch_size)
        ]
    else:
        train_dataset_batches = [train_dataset]
        test_dataset_batches = [test_dataset]

    model = LangModel(args)

    for idx, batch in tqdm(
        enumerate(train_dataset_batches),
        unit="batches",
        total=len(train_dataset_batches),
        desc="Generating train dataset",
    ):
        train_dataset_batches[idx] = model.generate_text(batch)

    for idx, batch in tqdm(
        enumerate(test_dataset_batches),
        unit="batches",
        total=len(test_dataset_batches),
        desc="Generating test dataset",
    ):
        test_dataset_batches[idx] = model.generate_text(batch)

    # DONE: Flatten the batches
    train_dataset = []
    test_dataset = []
    for batch in train_dataset_batches:
        train_dataset.extend(batch)

    for batch in test_dataset_batches:
        test_dataset.extend(batch)

    # DONE: Save the datasets
    if args.output_dataset_name is None:
        output_dataset_name = f"{JOB_TYPE}_{CURR_DATE_TIME}"
    else:
        output_dataset_name = args.output_dataset_name
    if not os.listdir("artifacts/" + output_dataset_name):
        os.makedirs("artifacts/" + output_dataset_name)
    with open(f"artifacts/{output_dataset_name}/train.json", "w") as f:
        json.dump(train_dataset, f)
    with open(f"artifacts/{output_dataset_name}/test.json", "w") as f:
        json.dump(test_dataset, f)

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
        "--batch-size",
        type=int,
        help="Number of prompts to process in a single inference",
    )
    parser.add_argument(
        "--train-test-split", type=float, default=0.8, help="Train test split ratio"
    )
    parser.add_argument(
        "--output-dataset-name",
        default=None,
        help="Output dataset name (default: current date and time)",
    )

    load_dotenv()

    main(parser.parse_args())
