#!/bin/bash

uv run lm_eval \
    --model vllm \
    --model_args pretrained=Qwen/Qwen3-0.6B,dtype=bfloat16,gpu_memory_utilization=0.95,think_end_token="</think>",max_gen_toks=4096 \
    --tasks ifbench \
    --apply_chat_template \
    --batch_size auto \
    -o results \
    -s