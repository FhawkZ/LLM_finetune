import torch
import json
import re
import string
import argparse
import os
import sys
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

F1_THRESHOLD = 0.6  
BATCH_SIZE = 4
MAX_NEW_TOKENS = 128 
TEMPERATURE = 0.7   
DO_SAMPLE = True  
TOP_K = 50        
TOP_P = 0.95        

def parse_args():
    parser = argparse.ArgumentParser(description="Generate negative samples.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local LLM .")
    parser.add_argument("--train_data_path", type=str, required=True, help="Input JSONL dataset path.")
    parser.add_argument("--output_neg_file", type=str, required=True, help="Output path for negative samples.")
    
    return parser.parse_args()

def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

# ------------------ NQ Style Metric Functions ------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def token_level_f1(prediction: str, ground_truths: list) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    max_f1 = 0.0
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        
        if normalized_prediction in ["yes", "no", "noanswer"] or normalized_ground_truth in ["yes", "no", "noanswer"]:
             if normalized_prediction == normalized_ground_truth:
                 return 1.0
             else:
                 continue
        pred_tokens = normalized_prediction.split()
        gt_tokens = normalized_ground_truth.split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        max_f1 = max(max_f1, f1)
    return max_f1

def is_acceptable(predicted_answer, correct_answer_list):
    """F1 分数 >= 阈值，则认为它是“可接受的”（即非负样本）。"""
    if not isinstance(correct_answer_list, list):
        correct_answer_list = [correct_answer_list]
        
    f1_score = token_level_f1(predicted_answer, correct_answer_list)
    return f1_score, f1_score >= F1_THRESHOLD

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() in self.stop_token_ids

# ------------------ 推理函数 ------------------
def get_answers(batch, model, tokenizer, device):
    prompts, targets = [], []
    for item in batch:
        instruction = item.get("instruction", "")
        input_doc = item.get("input", "")
        
        # 使用 Qwen 的 ChatML 格式来封装指令
        messages = [
            {"role": "system", "content": "You are a strict, concise answer extractor. Based ONLY on the documents provided, answer the question with the shortest possible answer extracted directly from the text. Your output must contain only the short answer and nothing else. If the answer is not in the document, respond ONLY with 'not found'."},
            {"role": "user", "content": f"Reference Documents:\n{input_doc}\n\nQuestion: {instruction}"}
        ]
        
        # 使用 apply_chat_template 构造 prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
        targets.append(item.get("output", []))

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    # 停止 token
    stop_tokens = [tokenizer.eos_token_id, tokenizer.encode(".")[0], tokenizer.encode("\n")[0]]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,        
            temperature=TEMPERATURE,    
            top_k=TOP_K,                
            top_p=TOP_P,
            num_return_sequences=1,     
            stopping_criteria=stopping_criteria,
            # 增加对 Qwen 特有 token 的检查，防止生成 system/user 标签
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>',)]
        )

    decoded_outputs = []
    for j in range(len(outputs)):

        full_text = tokenizer.decode(outputs[j], skip_special_tokens=False).strip()
      
        # 查找模型的回复开始标记 <|im_start|>assistant\n
        try:
            # 找到助手的回复开始位置
            assistant_start = full_text.rfind('<|im_start|>assistant\n')
            if assistant_start != -1:
                generated_text = full_text[assistant_start + len('<|im_start|>assistant\n'):]
            else:

                input_len_chars = len(prompts[j])
                generated_text = full_text[input_len_chars:]

            generated_text = generated_text.split('<|im_end|>')[0].strip()
            generated_text = generated_text.split('\n')[0].strip()
            generated_text = re.sub(r'Answer is:|Answer:|Short Answer:|not found|i don\'t know|the correct answer is', '', generated_text, flags=re.IGNORECASE).strip()

            if not generated_text:
                generated_text = "not found"
                
            decoded_outputs.append(generated_text)
            
        except Exception:
            decoded_outputs.append("Error in decoding.")           
    return decoded_outputs, targets

# ------------------ 主函数 ------------------
def main():
    args = parse_args()
    
    MODEL_PATH = args.model_path
    TRAIN_DATA_PATH = args.train_data_path
    OUTPUT_NEG_FILE = args.output_neg_file    
    print(f"Model Path: {MODEL_PATH}")
    print(f"Train Data: {TRAIN_DATA_PATH}")
    print(f"Output File: {OUTPUT_NEG_FILE}")

    # ------------------ 加载模型 ------------------
    print(f"Loading ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    try:
        data = read_jsonl(TRAIN_DATA_PATH)
        print(f"Dataset loaded. Total samples: {len(data)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_NEG_FILE), exist_ok=True)

    # ------------------ 负样本生成 (集成 F1 统计) ------------------
    neg_samples_count = 0
    acceptable_samples_count = 0 
    keys = ["instruction", "input", "output"]
    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with open(OUTPUT_NEG_FILE, "w", encoding="utf-8") as outfile:
        for i in tqdm(range(0, len(data), BATCH_SIZE), total=total_batches, desc="Processing Batches"):
            batch = data[i:i + BATCH_SIZE]
            batch_answers, targets = get_answers(batch, model, tokenizer, DEVICE)

            for item, pred_ans, target_ans in zip(batch, batch_answers, targets):
                
                if not isinstance(target_ans, list):
                    target_ans = [target_ans]
                f1_score, is_acceptable_bool = is_acceptable(pred_ans, target_ans)
                if is_acceptable_bool:
                    acceptable_samples_count += 1
                else:
                    neg_answer = pred_ans
                    neg_sample = {
                        "instruction": item.get(keys[0], ""),
                        "input": item.get(keys[1], ""),
                        "output": neg_answer,
                        "kto_tag": "false"
                    }
                    outfile.write(json.dumps(neg_sample, ensure_ascii=False) + '\n')
                    neg_samples_count += 1
            
            total_processed_in_batch = i + len(batch)
            if total_processed_in_batch > 0: 
                acceptance_rate = (acceptable_samples_count / total_processed_in_batch) * 100
                tqdm.write(f"\n--- Current F1 Performance (Processed: {total_processed_in_batch} samples) ---")
                tqdm.write(f"F1 >= {F1_THRESHOLD} (Acceptable/Correct): {acceptable_samples_count} ({acceptance_rate:.2f}%)")
                tqdm.write(f"F1 < {F1_THRESHOLD} (Negative Samples): {neg_samples_count}")
                tqdm.write("------------------------------------------------------------------")

    # 最终结果输出
    total_samples = len(data)
    acceptance_rate = (acceptable_samples_count / total_samples) * 100 if total_samples > 0 else 0
    print(f"\n Negative sample generation complete!")
    print(f"Total samples processed: {total_samples}")
    print(f"Total acceptable samples (F1 >= {F1_THRESHOLD}): {acceptable_samples_count} ({acceptance_rate:.2f}%)")
    print(f"Total negative samples generated: {neg_samples_count}")
    print(f"Output file: {OUTPUT_NEG_FILE}")

if __name__ == "__main__":
    main()