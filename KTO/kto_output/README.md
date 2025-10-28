---
library_name: peft
license: other
base_model: /root/autodl-tmp/Qwen3-8B-AWQ
tags:
- base_model:adapter:/root/autodl-tmp/Qwen3-8B-AWQ
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: train_2025-10-28-10-08-54
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2025-10-28-10-08-54

This model is a fine-tuned version of [/root/autodl-tmp/Qwen3-8B-AWQ](https://huggingface.co//root/autodl-tmp/Qwen3-8B-AWQ) on the kto_dataset_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4615
- Rewards/chosen: 0.5192
- Logps/chosen: -66.5088
- Logits/chosen: 20778346.1463
- Rewards/rejected: 0.2186
- Logps/rejected: -83.1894
- Logits/rejected: 8759033.4359
- Rewards/margins: 0.3006
- Kl: 6.8895
- Num Input Tokens Seen: 6388096

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-07
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Rewards/chosen | Logps/chosen | Logits/chosen | Rewards/rejected | Logps/rejected | Logits/rejected | Rewards/margins |        | Input Tokens Seen |
|:-------------:|:------:|:----:|:---------------:|:--------------:|:------------:|:-------------:|:----------------:|:--------------:|:---------------:|:---------------:|:------:|:-----------------:|
| 0.4848        | 0.4040 | 100  | 0.4705          | 0.3227         | -66.9017     | 21281509.4634 | 0.0805           | -83.4656       | 9404417.6410    | 0.2423          | 5.3726 | 2584128           |
| 0.4595        | 0.8081 | 200  | 0.4594          | 0.5385         | -66.4702     | 20884344.1951 | 0.1943           | -83.2380       | 8775205.7436    | 0.3442          | 7.1505 | 5169952           |


### Framework versions

- PEFT 0.17.1
- Transformers 4.51.3
- Pytorch 2.7.0+cu128
- Datasets 4.0.0
- Tokenizers 0.21.4