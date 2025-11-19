import os
import json
import argparse
from tqdm import tqdm

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM

from metrics import sub_exact_match, dataset_level_f1, batch_sub_exact_match


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QA with LLaMA model and permuted docs, and save results."
    )
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--input",type=str,required=True)
    parser.add_argument("--output",type=str,required=True)

    return parser.parse_args()


def load_qwen(model_name):
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def load_qa_documents(data_path):
    data = []
    if not os.path.exists(data_path):
        print(f"Error: Input file not found at {data_path}")
        return [], [], [], []

    with open(data_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                d = json.loads(line)
                data.append(d)
            except Exception as e:
                print(f"Error loading line {line_no}: {e}. Content preview: {line[:200]!r}")

    q = [d.get("question", "") for d in data]
    a = [d.get("answer", "") for d in data]
    docs = [d.get("documents", []) for d in data]

    return q, a, docs, data


def get_answer(documents, question, model, tokenizer):
    final_docs = "\n".join(
        [f"document {j}: {doc}" for j, doc in enumerate(documents, 1)]
    )
    system_prompt = ""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Documents:\nQuestions:{question}\n{final_docs}\nAnswer(only the short phrase) :",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion (加上 no_grad/inference_mode 更高效)
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32,
            temperature=0.01,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:

        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # 只打印最终的 content，不打印 documents
    print("content:", content)
    return content


def main():
    args = parse_args()
    question, answer, documents, raw_data = load_qa_documents(args.input)

    if not raw_data:
        print("Exiting due to data loading failure.")
        return

    model, tokenizer = load_qwen(args.model)

    number = 0
    answer_set = []

    with open(args.output, "w", encoding="utf-8") as outfile:
        print(f"Results will be saved to: {args.output}")

        for i in tqdm(range(len(question))):
            ans = get_answer(
                documents=documents[i],
                question=question[i],
                model=model,
                tokenizer=tokenizer,
            )

            cleaned_ans = ans.strip().rstrip(".")
            answer_set.append(cleaned_ans)

            current_em = sub_exact_match(ans, answer[i])
            number += current_em
            current_em_rate = number / (i + 1)

            avg_f1, _ = dataset_level_f1(answer_set, answer[0: len(answer_set)])

            print(f"Current EM Rate: {current_em_rate:.4f}")
            print(f"Current F1: {avg_f1:.4f}")

            result_entry = raw_data[i].copy()
            result_entry["predicted_answer"] = cleaned_ans
            result_entry["em_score"] = current_em

            result_entry.pop("documents", None)

            outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
            outfile.flush()

        avg_score, _ = batch_sub_exact_match(answer_set, answer)
        print("-" * 30)
        print(f"Final EM: {avg_score:.4f}")
        avg_f1, _ = dataset_level_f1(answer_set, answer)
        print(f"Final F1: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
