#!/bin/bash

# uv sync --extra vllm,ifbench,ifeval,hf

uv run lm_eval \
    --model vllm \
    --model_args \
        pretrained=Qwen/Qwen3.5-2B \
        dtype=bfloat16 \
        gpu_memory_utilization=0.95 \
        think_end_token="</think>" \
        max_gen_toks=32000 \
    --tasks ifeval,ifbench,aime25,aime26,hendrycks_math500,humaneval_instruct,humaneval_plus_instruct,mbpp_instruct,mbpp_plus_instruct \
    --gen_kwargs \
        temperature=1.0 \
        top_p=0.95 \
        top_k=20 \
        min_p=0.0 \
        presence_penalty=1.5 \
        repetition_penalty=1.0 \
        max_gen_toks=32000 \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size auto \
    --confirm_run_unsafe_code \
    -o results