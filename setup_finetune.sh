#!/bin/bash

# Create directories
mkdir -p finetune_data/chat
mkdir -p finetune_data/images/coco/train2017
mkdir -p finetune_data/images/gqa/images
mkdir -p finetune_data/images/ocr_vqa/images
mkdir -p finetune_data/images/textvqa/train_images
mkdir -p finetune_data/images/vg/VG_100K
mkdir -p finetune_data/images/vg/VG_100K_2

# Download datasets
wget -P finetune_data/chat/ https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json
wget -P finetune_data/images/coco/train2017/ http://images.cocodataset.org/zips/train2017.zip
wget -P finetune_data/images/gqa/images/ https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
wget -P finetune_data/images/vg/VG_100K_2/ https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
wget -P finetune_data/images/vg/VG_100K/ https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -P finetune_data/images/textvqa/train_images/ https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

# Unzip datasets
unzip finetune_data/images/coco/train2017/train2017.zip -d finetune_data/images/coco/train2017/
unzip finetune_data/images/gqa/images/images.zip -d finetune_data/images/gqa/images/
unzip finetune_data/images/vg/VG_100K_2/images2.zip -d finetune_data/images/vg/VG_100K_2/
unzip finetune_data/images/vg/VG_100K/images.zip -d finetune_data/images/vg/VG_100K/
unzip finetune_data/images/textvqa/train_images/train_val_images.zip -d finetune_data/images/textvqa/train_images/

# Remove zip files
rm finetune_data/images/coco/train2017/train2017.zip
rm finetune_data/images/gqa/images/images.zip
rm finetune_data/images/vg/VG_100K_2/images2.zip
rm finetune_data/images/vg/VG_100K/images.zip
rm finetune_data/images/textvqa/train_images/train_val_images.zip

# Download Python script and dataset.json from Google Drive
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=16eqkNbgARX1aLM4q0l5WBiPPSFbK0Elp' -O download_ocr_vqa.py
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1r0tyZUwGCc4wIG4RkiglCGNL_nFJjR6Q' -O dataset.json

# Run the Python script to download OCR-VQA data
python download_ocr_vqa.py
