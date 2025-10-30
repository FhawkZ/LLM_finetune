from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import re
import argparse
# 加载量化模型
# tensorboard 可视化
from transformers import StoppingCriteria, StoppingCriteriaList,AutoModelForCausalLM



#====================================================参数区====================================================
parser = argparse.ArgumentParser(description="Train GRPO model with LoRA and AWQ")
parser.add_argument("--use_documents", type=int, default=1, help="是否在提示中包含文档内容，1: True, 0: False")
parser.add_argument("--model_name", type=str, default="Official_LLMs/Qwen3-8B-AWQ", help="模型名称或路径")
parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="LoRA target modules, 用逗号分隔")
parser.add_argument("--dataset_path", type=str, default="Dataset/nq/train_top5.json", help="数据集路径")
parser.add_argument("--device_map", type=str, default="cuda:1", help="模型加载设备")
parser.add_argument("--torch_dtype", type=str, default="float16", help="模型数据类型")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--output_dir", type=str, default="RL/TheoryandApplication/output", help="训练输出路径")
parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
parser.add_argument("--num_generations", type=int, default=4, help="每轮生成数量")
parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每卡训练批量大小")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率")
parser.add_argument("--fp16", type=int, default=1, help="是否启用 fp16，1: True, 0: False")
parser.add_argument("--use_vllm", type=int, default=0, help="是否使用 vLLM 加速生成，1: True, 0: False")
args = parser.parse_args()

print("训练参数:", args)

use_documents = bool(args.use_documents)
Target_modules = args.target_modules.split(",")
Torch_dtype = getattr(torch, args.torch_dtype)


# —— 在构建 trainer 之前调用一次 —— 
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map=args.device_map,
    trust_remote_code=True,
    torch_dtype=Torch_dtype,#LoRA adapter 仍然是 float16 参数类型
)


#====================================================定义模型配置====================================================

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=Target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(model)

dataset = load_dataset("json", data_files=args.dataset_path, split="train")
# 只保留前10000条数据（如果总样本数不足10000，会自动取全部）
dataset = dataset.select(range(min(10000, len(dataset))))


    
def format_prompts(examples):
    prompts = []
    answers = []
    # examples["documents"]是一个整个batch的列表
    for doc, q, a in zip(examples["documents"], examples["question"], examples["answer"]):
        if use_documents:
            prompt = f"Documents: {doc}\nQuestion: {q}\nPlease provide only the final answer, wrapped in <answer> tags.\nAnswer:"
            # prompt = f"Based on the provided reference documents, answer the following question:{q}\n{doc}\nPlease provide only the final answer, wrapped in <answer> tags.\nAnswer:"
        else:
            prompt = f"Question: {q}\nPlease provide only the final answer, wrapped in <answer> tags.\nAnswer:"
        prompts.append(prompt)
        answers.append(a)  # ✅ 这里是一对一

    return {"prompt": prompts, "answer": answers}

#Dataset.map() v在 batched=True 时，会自动将数据划分为多个“批次（batch）”，并一次性将整个 batch 传给你定义的函数 format_prompts()
dataset = dataset.map(format_prompts, batched=True)


#====================================================定义奖励函数====================================================
def extract_final_answer(completion: str):
    # 正则匹配 <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)

    if match:
        answer = match.group(1).strip()
        return answer

    return None

    
def reward_len(completions, **kwargs):
    ideal_length = 20
    return [-abs(ideal_length - len(completion)) for completion in completions]


def reward_format(completions, **kwargs):

    pattern = r"^\s*<answer>.*?</answer>\s*"
    return [1.0 if re.match(pattern, c) else 0.0 for c in completions]



def problem_reward(completions, answers, **kwargs):
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        answer = extract_final_answer(completion)
        if answer is None:
            rewards.append(0.0)
            continue
        # for correct in correct_answer:
        #     if ...:
        #         rewards.append(1.0)
        #     else:
        #         rewards.append(0.0)
        #     break  这里每次循环都会 break，其实只检查了第一个答案

        # correct_answer 可能是列表
        if any(str(answer).strip().lower() == str(correct).strip().lower() for correct in correct_answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def make_reward_function(dataset):
    def reward_func(prompts, completions, completion_ids, **batch):
        """
        创建奖励函数，融合格式正确性和问题解决正确性奖励

        Args:
            completions: 模型生成的完成结果列表
            batch: 包含答案等信息的批次数据
            **kwargs: 其他关键字参数

        Returns:
            list[float]: 融合后的标量奖励值列表
        """
        answers = batch.get("answer", {})

        print(f"completions:{completions}")
        print(f"answers:{answers}")
        print("====================================================")
        """融合长度、格式、正确性奖励"""
        # 各子奖励
        # len_rewards = reward_len(completions)
        fmt_rewards = reward_format(completions)
        correctness_rewards = problem_reward(completions, answers or [""] * len(completions))
        print(f"fmt_rewards:{fmt_rewards}")
        print(f"correctness_rewards:{correctness_rewards}")

        # 加权系数（可以调）
        # w_len = 0.2
        w_fmt = 0.4
        w_correct = 0.6

        # 融合为一个标量奖励
        final_rewards = []
        for f, c in zip(fmt_rewards, correctness_rewards):
            final_rewards.append(w_fmt * f + w_correct * c)

        print(f"final_rewards:{final_rewards}")
        print("====================================================")
        return final_rewards
    return reward_func


#====================================================训练====================================================
training_args = GRPOConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    num_generations=args.num_generations,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    fp16=bool(args.fp16),
    logging_steps=1,
    report_to="tensorboard",
    use_vllm=bool(args.use_vllm),

    generation_kwargs = {
        "temperature": 0.7,   # 增加grpo训练时采样随机性
        "top_p": 0.9,
        "top_k": 20,
        "do_sample": True,
        "max_new_tokens": 32,
    },
    save_steps=500,  
    # 可选：限制最多保存的模型数量（防止磁盘占满）
    save_total_limit=11,  
)
# 6. 初始化GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=make_reward_function(dataset),
)

# 7. 开始训练
trainer.train()