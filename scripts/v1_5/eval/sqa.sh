#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path SkunkworksAI/BakLLaVA-1 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/BakLLaVA-1_2.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/BakLLaVA-1_2.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/BakLLaVA-1_2_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/BakLLaVA-1_2_result.json
