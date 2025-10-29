---
library_name: peft
license: other
base_model: /media/data/users/liqz/Qwen/Qwen3-8B-AWQ
tags:
- base_model:adapter:/media/data/users/liqz/Qwen/Qwen3-8B-AWQ
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: train_lr_1e-6
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_lr_1e-6

This model is a fine-tuned version of [/media/data/users/liqz/Qwen/Qwen3-8B-AWQ](https://huggingface.co//media/data/users/liqz/Qwen/Qwen3-8B-AWQ) on the nq_top5_train dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0158
- Num Input Tokens Seen: 61455152

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-06
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 1.0

### Training results



### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.9.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.1