import torch
import wandb
import argparse
import datasets
import numpy as np
import transformers
from trl import SFTTrainer
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel
from torchmetrics.text.rouge import ROUGEScore


def calc_metrics(inputs: list, outputs: list):
    try:
        rouge_r = ROUGEScore()
        return rouge_r(outputs, inputs)
    except:
        rouge_r = ROUGEScore(rouge_keys=("rouge1", "rouge2"))
        return rouge_r(outputs, inputs)


def doTest(model, tokenizer, testds):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = pipeline(
        testds,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return [
        outputs[i][0]["generated_text"][len(testds[i]) :] for i in range(len(testds))
    ]


def main(args):

    wandb.init(
        entity="jhu-llm-prompt-recovery",
        project="llm-prompt-recovery",
        job_type="qlora",
        name=f'qlora-{args.dataset_subset.split("-")[0]}',
    )

    dataset = datasets.load_dataset(args.dataset, args.dataset_subset)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_size = "left"

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "all-linear"],
    )

    peft_model = PeftModel.from_pretrained(
        base_model,
        args.model,
        subfolder="loftq_init",
        is_trainable=True,
        config=peft_config,
    )

    # Initializing TrainingArguments with default values
    training_args = transformers.TrainingArguments(
        output_dir="./trained/",
        evaluation_strategy="epoch",
        do_eval=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        hub_model_id=f"nrishabh/llama3-8b-instruct-qlora-{args.dataset_subset.split('-')[0]}",
        learning_rate=args.learning_rate,
        log_level="info",
        logging_steps=1,
        logging_strategy="epoch",
        lr_scheduler_type="cosine",
        max_steps=-1,
        num_train_epochs=args.epochs,
        disable_tqdm=False,
        dataloader_pin_memory=True,
        report_to="wandb",  # for skipping wandb logging
        save_strategy="no",
        seed=42,
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    torch.cuda.empty_cache()

    train_result = trainer.train()

    trainer.push_to_hub(token="")

    target = dataset["test"]["completion"]
    baseline = doTest(peft_model, tokenizer, dataset["test"]["prompt"])
    finetuned = doTest(trainer.model, tokenizer, dataset["test"]["prompt"])
    baseline_metrics = calc_metrics(target, baseline)
    finetuned_metrics = calc_metrics(target, finetuned)

    wandb.log({"target": target, "baseline": baseline, "finetuned": finetuned})
    wandb.log({"baseline_rouge": baseline_metrics})
    wandb.log({"finetuned_metrics": finetuned_metrics})

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Lora Model")
    parser.add_argument(
        "--dataset", type=str, default="nrishabh/prompt-recovery", help="Dataset name"
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="minute-llama-instr",
        help="Dataset subset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LoftQ/Meta-Llama-3-8B-Instruct-4bit-64rank",
        help="Model name",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Evaluation batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")

    load_dotenv("/home/rnanawa1/llm-prompt-recovery/.env")

    args = parser.parse_args()
