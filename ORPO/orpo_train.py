import os
import random
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import ORPOConfig, ORPOTrainer
from peft import LoraConfig
from utils import *


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer and model (this may take a while)...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    raw_train = read_jsonl(args.train_file)
    raw_train = raw_train[:10000]

    print(f"Read {len(raw_train)} train records")
    print("Building ORPO-style datasets (creating synthetic 'rejected' responses)...")
    train_ds = build_orpo_dataset(raw_train, tokenizer, max_length=args.max_length, rng=random.Random(args.seed))

    orpo_config = ORPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        
        learning_rate=2e-5,
        save_steps=1000,

        max_grad_norm=1.0,
        weight_decay=0.01,
        seed=args.seed,

        logging_steps=10,
        logging_strategy="steps",
        logging_dir="./log",
        report_to="tensorboard"
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer, 
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model to output_dir...")
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="path-to-model", help="Path or HF id for Qwen3-8B-AWQ checkpoint")
    parser.add_argument("--train_file", type=str, default="path-to-train-data", help="train.jsonl path")
    parser.add_argument("--test_file", type=str, default="path-to-test-data", help="test.jsonl path")
    parser.add_argument("--output_dir", type=str, default="path-to-save-model")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)


