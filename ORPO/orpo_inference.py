import torch
import random
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils import *


def inference(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.output_dir)
    model.eval()

    # For generation, move model to a single device if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    raw_test = read_jsonl(args.test_file)
    # raw_test = raw_test[:100]

    answers = []
    golds = []
    batch_size = 64
    for i in tqdm(range(0, len(raw_test), batch_size), total=(len(raw_test) + batch_size - 1)//batch_size):
        batch = raw_test[i:i + batch_size]
        batch_answers = get_answers(batch, model, tokenizer)

        for ans in batch_answers:
            if "response: " in ans:
                ans = ans[ans.find("response: "):]
            answers.append(ans.strip())
        
        golds.extend([item["answer"] for item in batch])
        
    with open("answer_.json", "w") as f:
        json.dump(answers, f, indent=4)
    
    # with open("answer_.json", "r") as f:
    #     answers = json.load(f)
        
    batch_sub_exact_match_score = batch_sub_exact_match(answers, golds)
    dataset_level_f1_score = dataset_level_f1(answers, golds)
    with open("result_.json", "w") as f:
        json.dump({
            "batch_sub_exact_match_score": batch_sub_exact_match_score,
            "dataset_level_f1_score": dataset_level_f1_score,
        }, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/liangzhilin/model/qwen3-8b-awq", help="Path or HF id for Qwen3-8B-AWQ checkpoint")
    parser.add_argument("--train_file", type=str, default="/home/liangzhilin/workspace/25AICourse/train_top5.json", help="train.jsonl path")
    parser.add_argument("--test_file", type=str, default="/home/liangzhilin/workspace/25AICourse/test_top5.json", help="test.jsonl path")
    parser.add_argument("--output_dir", type=str, default="/home/liangzhilin/orpo_output_")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inference(args)