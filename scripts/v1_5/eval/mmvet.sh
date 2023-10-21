#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path SkunkworksAI/BakLLaVA-1\
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/BakLLaVA-1_2.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/BakLLaVA-1_2.jsonl \
    --dst ./playground/data/eval/mm-vet/results/BakLLaVA-1_2.json

