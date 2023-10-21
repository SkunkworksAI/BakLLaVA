#!/bin/bash
sh /data/lychen/code/key.sh

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
--model-path SkunkworksAI/BakLLaVA-1 \
--question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
--answers-file ./playground/data/eval/mmbench/answers/$SPLIT/BakLLaVA-1_2.jsonl \
--single-pred-prompt \
--temperature 0 \
--conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment BakLLaVA-1_2


python /data/lychen/code/decoding/opencompass/tools/eval_mmbench.py /data/lychen/code/decoding/LLaVA/playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712/BakLLaVA-1_2.xlsx --meta /data/lychen/code/decoding/LLaVA/playground/data/eval/mmbench/mmbench_dev_20230712.tsv