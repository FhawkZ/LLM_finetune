# streamlit_compare6_qwen_jsonl_enable_sidebar.py
import os
import re
import gc
import json
import threading
from typing import Dict, Generator, Iterable, List, Tuple, Union, Optional
from difflib import get_close_matches

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

# ===================== 固定配置（可按需改为你机器上的路径） =====================
BASE_MODEL_DIR = "/root/.ollama/Qwen3-8B-AWQ"
ONLY_LOCAL = True  # 离线加载（如需在线下载改 False）
# SYSTEM_PROMPT = (
#     "You are a helpful assistant. Use the provided documents if any. "
#     "Reply directly with the final answer. Do NOT output <think>…</think>."
# )
SYSTEM_PROMPT = ("")
DATASET_PATH = "/home/course/ai_course/scream_list/dual-llm-chat-playground/adapter/test_top5.json"  # 固定 JSONL 路径
DOC_TOP_K = 3            # 组装进 prompt 的文档条数
STREAM_TO_PAGE = True    # 使用流式写入页面
STRIP_THINK_DISPLAY = True  # 显示层去除 <think>…</think>

# 适配器路径（保留功能；侧栏不展示，直接用固定路径）
ADAPTER_ROOT = "/home/course/ai_course/scream_list/dual-llm-chat-playground/adapter"
ADAPTER_PATHS = {
    "SFT":  os.path.join(ADAPTER_ROOT, "SFT"),
    "DPO":  os.path.join(ADAPTER_ROOT, "DPO"),
    "ORPO": os.path.join(ADAPTER_ROOT, "ORPO"),
    "KTO":  os.path.join(ADAPTER_ROOT, "KTO"),
    "GRPO": os.path.join(ADAPTER_ROOT, "GRPO"),
}

# ===================== 页面设置 =====================
st.set_page_config(page_title="Qwen 6× Compare + JSONL", layout="wide")
st.title("✨ Display of Large Langugae Model Fine-Tuning")

# ===================== 侧边栏（只保留采样3项 + 每模型 enable_* 开关） =====================
with st.sidebar:
    st.subheader("Generation")
    max_new_tokens = st.number_input("max_new_tokens", 16, 4096, 256, 16)
    temperature    = st.slider("temperature", 0.0, 1.5, 0.7, 0.05)
    top_p          = st.slider("top_p", 0.0, 1.0, 0.95, 0.05)

    st.markdown("---")
    st.subheader("Enable models")
    enable_Base = st.checkbox("enable_Base", True)
    enable_SFT  = st.checkbox("enable_SFT",  True)
    enable_DPO  = st.checkbox("enable_DPO",  True)
    enable_ORPO = st.checkbox("enable_ORPO", True)
    enable_KTO  = st.checkbox("enable_KTO",  True)
    enable_GRPO = st.checkbox("enable_GRPO", True)

# ===================== 基础工具函数 =====================
@st.cache_resource(show_spinner=False)
def load_base_model_and_tokenizer(model_id: str, local_only: bool):
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=local_only, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return mdl, tok

def _parse_jsonl_lines(lines: Iterable[str]) -> List[Dict]:
    records: List[Dict] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except Exception as e:
            st.warning(f"JSONL parse error at line {i+1}: {e}")
    return records

@st.cache_data(show_spinner=False)
def load_jsonl_from_path(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _parse_jsonl_lines(f.readlines())

def normalize_q(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def find_by_question(samples: List[Dict], q: str) -> Tuple[Optional[int], Optional[Dict], str]:
    """先精确匹配（忽略大小写/空白），找不到再模糊匹配。"""
    if not samples:
        return None, None, "none"
    nq = normalize_q(q)
    for i, r in enumerate(samples):
        if "question" in r and normalize_q(str(r["question"])) == nq:
            return i, r, "exact"
    pool = [str(r.get("question", "")) for r in samples]
    cand = get_close_matches(q, pool, n=1, cutoff=0.85)
    if cand:
        idx = pool.index(cand[0])
        return idx, samples[idx], "fuzzy"
    return None, None, "none"

def make_prompt_from_record(question: str, documents: List[str], k: int) -> Tuple[str, List[str]]:
    
    print("wwwwwww",documents,"\n\n\n")
    if not documents:
        return question, []
    k = max(1, min(k, len(documents)))
    # picked = documents[:k]
    picked = documents

    print("kkkkkk",picked,"\n\n\n")

    # ctx = "\n\n".join([f"[Doc {i+1}] {d}" for i, d in enumerate(picked)])
    ctx = documents
    user_block = (
        f"Based on the provided reference documents, answer the following question with only the result, no extra text:{question}\n"
        f"{ctx}"
    )
    return user_block, picked

def make_grpo_user_text(question: str, documents: List[str]) -> str:
    # 把文档拼起来；没有文档时给空串也可以
    docs_str = "\n\n".join(documents) if documents else ""
    return (
        f"Documents:{docs_str}\n"
        f"Question:Based on the provided reference documents, answer the following question with only the result, no extra text:{question}\n\n"
    )


def build_inputs(tok, user_text: str, system_text: str = "") -> Dict[str, torch.Tensor]:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    print("messages",messages,"\n\n\n")
    # 对部分 tokenizer 兼容（有的支持 enable_thinking，有的没有）
    try:
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok([prompt_text], return_tensors="pt")

def finalize_kwargs(tok, max_new_tokens: int, temperature: float, top_p: float) -> Dict:
    return dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

def generate_stream(model, tok, enc_inputs: Dict[str, torch.Tensor], **gen_kw) -> Generator[str, None, None]:
    inputs = {k: v.to(model.device) for k, v in enc_inputs.items()}
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    th = threading.Thread(target=model.generate, kwargs=dict(**inputs, streamer=streamer, **gen_kw), daemon=True)
    th.start()
    for piece in streamer:
        yield piece
    th.join()

def generate_once(model, tok, enc_inputs: Dict[str, torch.Tensor], **gen_kw) -> str:
    inputs = {k: v.to(model.device) for k, v in enc_inputs.items()}
    out = model.generate(**inputs, **gen_kw)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

def strip_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _adapter_is_valid(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))

def peft_for_variant(base_model, adapter_dir: str, adapter_name: str) -> PeftModel:
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, adapter_name=adapter_name, is_trainable=False)
    try:
        peft_model.set_adapter(adapter_name)
    except Exception:
        pass
    return peft_model

def render_stream_or_text(title: str, gen_iter: Union[Iterable[str], str], stream: bool = True):
    st.subheader(title)

    # 插入 CSS 样式：控制 model 输出区字体大小与 “Your question” 一致
    st.markdown(
        """
        <style>
        .model-output {
            font-size:22px !important;
            font-weight:400 !important;
            color:#333333 !important;
            line-height:1.4 !important;
            font-family: "Helvetica Neue", Arial, sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if isinstance(gen_iter, str) or not stream:
        # 静态输出分支
        text = gen_iter if isinstance(gen_iter, str) else "".join(list(gen_iter))
        if STRIP_THINK_DISPLAY and text:
            text = strip_think_block(text)
        st.markdown(f"<div class='model-output'>{text}</div>", unsafe_allow_html=True)
    else:
        # 流式输出分支
        # 将流式片段合并然后统一包裹
        buffered = "".join(list(gen_iter))
        if STRIP_THINK_DISPLAY and buffered:
            buffered = strip_think_block(buffered)
        st.markdown(f"<div class='model-output'>{buffered}</div>", unsafe_allow_html=True)


# ===================== 用户输入（问题） =====================
query = st.chat_input("Type your question here and press Enter")
if not query:
    st.stop()


# 顶部展示“你输入的问题原文”
with st.container(border=True):
    # st.markdown(f"**Your question:** {query}")
    st.markdown(
        f"<p style='font-size:22px;'><span style='font-weight:600;'>Your question:</span> {query}</p>",
        unsafe_allow_html=True
    )

# ===================== 读取固定 JSONL 并检索 =====================
records: List[Dict] = load_jsonl_from_path(DATASET_PATH)
match_idx, matched, mode = find_by_question(records, query) if records else (None, None, "none")

# prefix = "Based on the provided reference documents, answer the following question with only the result, no extra text:"
# query = prefix + query

if matched:
    user_prompt_text, picked_docs = make_prompt_from_record(
        matched.get("question", query), matched.get("documents", []), DOC_TOP_K
    )
else:
    user_prompt_text, picked_docs = query, []

print("ssssss",user_prompt_text,"\n\n\n")
print("qqqqqq",picked_docs,"\n\n\n")

with st.container(border=True):
    if matched:
        if picked_docs:
            st.markdown("**Document #1 (preview):**")
            st.write(picked_docs[0])
        if "answer" in matched:
            st.markdown("**Gold Answer(s):**")
            st.write(matched["answer"])
            # st.write(matched["answer"][0])
    else:
        st.info("No dataset match. Model will use your question only.")

# ===================== 静默加载基座（不显示“Base loaded”） =====================
try:
    base_model, tok = load_base_model_and_tokenizer(BASE_MODEL_DIR, ONLY_LOCAL)
except Exception as e:
    st.error(f"❌ Failed to load base model: {e}")
    st.stop()

enc  = build_inputs(tok, user_prompt_text, SYSTEM_PROMPT)
genk = finalize_kwargs(tok, max_new_tokens, temperature, top_p)

# ===================== 渲染勾选启用的模型 =====================
def pane_title(name: str) -> str:
    return "Qwen3-8B-AWQ (Base)" if name == "Base" else name

def run_base():
    return generate_stream(base_model, tok, enc, **genk)

def run_adapter(name: str):
    path = ADAPTER_PATHS.get(name, "")
    if not _adapter_is_valid(path):
        st.error(f"{name}: adapter invalid or not found -> {path} (need adapter_config.json)")
        return None
    try:
        peft_mdl = peft_for_variant(base_model, path, adapter_name=name)

        # ====== 仅 GRPO 使用独立提示词 ======
        if name == "GRPO":
            q_for_grpo = matched.get("question", query) if matched else query
            docs_for_grpo = picked_docs if picked_docs else []
            grpo_user_text = make_grpo_user_text(q_for_grpo, docs_for_grpo)
            enc_local = build_inputs(tok, grpo_user_text, system_text="")  # 不传 system 提示
        else:
            enc_local = enc  # 其他模型用默认提示词
        # ==================================

        # gen = generate_stream(peft_mdl, tok, enc, **genk)
        gen = generate_stream(peft_mdl, tok, enc_local, **genk)
        return gen
    except Exception as e:
        st.error(f"{name}: load/use adapter failed -> {e}")
        return None
    finally:
        try:
            del peft_mdl
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 按侧栏 enable_* 开关决定哪些窗格启用
enabled_names = []
if enable_Base: enabled_names.append("Base")
if enable_SFT:  enabled_names.append("SFT")
if enable_DPO:  enabled_names.append("DPO")
if enable_ORPO: enabled_names.append("ORPO")
if enable_KTO:  enabled_names.append("KTO")
if enable_GRPO: enabled_names.append("GRPO")

# 三列网格布局
rows = []
for i in range(0, len(enabled_names), 3):
    rows.append(st.columns(min(3, len(enabled_names) - i)))

idx = 0
for row in rows:
    for cell in row:
        name = enabled_names[idx]
        with cell.container(border=True):
            if name == "Base":
                gen = run_base()
            else:
                gen = run_adapter(name)
            if gen is not None:
                render_stream_or_text(pane_title(name), gen, stream=STREAM_TO_PAGE)
        idx += 1
