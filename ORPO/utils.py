import torch
import random
import json
from functools import partial
from datasets import Dataset, load_dataset, concatenate_datasets
from utils import *

USER_PROMPT = "Based on the provided reference documents, answer the following question:{question}\
query: {document}"
ASSISTANT_PROMPT = "response: {answer}"

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def build_orpo_dataset(raw, tokenizer, max_length=1024, rng=None):

    rng = rng or random.Random()

    samples = []
    answers_pool = [ (i, item['answer'][0] if item.get('answer') else "") for i,item in enumerate(raw) ]

    for idx, item in enumerate(raw):
        q = item.get('question','')
        answers = item.get('answer', [])
        documents = item.get('documents', [])
        chosen = answers[0] if answers else ''
        doc_text = "\n\n".join(documents) if documents else ""
        prompt = USER_PROMPT.format(question=q, document=doc_text)

        rejected = ''
        attempts = 0
        while attempts < 5:
            ridx, other_ans = rng.choice(answers_pool)
            if ridx != idx and other_ans.strip():
                rejected = other_ans
                break
            attempts += 1
        if not rejected:
            rejected = "I don't know."  

        samples.append({"prompt": prompt, "chosen": ASSISTANT_PROMPT.format(answer=chosen), "rejected": ASSISTANT_PROMPT.format(answer=rejected)})

    ds = Dataset.from_list(samples)
    return ds


def get_answers(batch_examples, model, tokenizer):
    batch_texts = []
    for ex in batch_examples:
        q = ex.get("question", "")
        docs = ex.get("documents", [])
        final_docs = "\n".join([doc for doc in docs])
        prompt = USER_PROMPT.format(question=q, document=final_docs)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        batch_texts.append(text)

    model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32,
            temperature=0.01,
            top_p=1.0,
            top_k=50,
            do_sample=True
        )

    answers = []
    for i in range(len(batch_examples)):
        input_len = len(model_inputs.input_ids[i])
        output_ids = generated_ids[i][input_len:].tolist()
        
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        answers.append(content)

    return answers


import re
import string
from collections import Counter


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


def sub_exact_match(prediction: str, golden_answers) -> float:

    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    score = 0.0

    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1.0
            break

    return score


def batch_sub_exact_match(
    pred_list,
    golden_answers_list,
):
    score_list = [
        sub_exact_match(pred, golden_answers)
        for pred, golden_answers in zip(pred_list, golden_answers_list)
    ]
    avg_score = sum(score_list) / len(score_list)
    return avg_score, score_list


def token_level_f1(prediction: str, ground_truths: list):

    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)

        if (
            normalized_prediction in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            continue
        if (
            normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
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

        final_metric["precision"] = max(final_metric["precision"], precision)
        final_metric["recall"] = max(final_metric["recall"], recall)
        final_metric["f1"] = max(final_metric["f1"], f1)

    return final_metric


def dataset_level_f1(pred_list: list, golden_answers_list: list):

    metric_scores = []
    for pred, golds in zip(pred_list, golden_answers_list):
        score = token_level_f1(pred, golds)["f1"]
        metric_scores.append(score)

    avg_f1 = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
    return avg_f1, metric_scores
