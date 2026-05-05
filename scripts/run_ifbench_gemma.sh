#!/bin/bash

# uv sync --extra vllm,ifbench,ifeval,hf

uv run lm_eval \
    --model vllm \
    --model_args \
        pretrained=google/gemma-4-E2B-it \
        dtype=bfloat16 \
        gpu_memory_utilization=0.95 \
        think_end_token="<channel|>" \
        max_gen_toks=32000 \
    --tasks ifeval,ifbench,aime25,aime26,hendrycks_math500,humaneval_instruct,humaneval_plus_instruct,mbpp_instruct,mbpp_plus_instruct \
    --gen_kwargs \
        temperature=1.0 \
        top_p=0.95 \
        top_k=64 \
        max_gen_toks=32000 \
    --apply_chat_template \
    --limit 2 \
    --num_fewshot 0 \
    --batch_size auto \
    --confirm_run_unsafe_code \
    -o results \
    -s