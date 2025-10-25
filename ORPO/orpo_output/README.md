---
library_name: peft
model_name: orpo_output
tags:
- base_model:adapter:/home/liangzhilin/model/qwen3-8b-awq
- lora
- orpo
- transformers
- trl
licence: license
base_model: /home/liangzhilin/model/qwen3-8b-awq
pipeline_tag: text-generation
---

# Model Card for orpo_output

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with ORPO, a method introduced in [ORPO: Monolithic Preference Optimization without Reference Model](https://huggingface.co/papers/2403.07691).

### Framework versions

- PEFT 0.17.1
- TRL: 0.24.0
- Transformers: 4.51.3
- Pytorch: 2.8.0
- Datasets: 4.3.0
- Tokenizers: 0.21.4

## Citations

Cite ORPO as:

```bibtex
@article{hong2024orpo,
    title        = {{ORPO: Monolithic Preference Optimization without Reference Model}},
    author       = {Jiwoo Hong and Noah Lee and James Thorne},
    year         = 2024,
    eprint       = {arXiv:2403.07691}
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```