#!/bin/bash

# Create directories
mkdir -p pretrain_data/chat
mkdir -p pretrain_data/images

# Download datasets
wget -P pretrain_data/chat/ https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json
wget -P pretrain_data/images/ https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip

# Unzip datasets
unzip pretrain_data/images/images.zip -d pretrain_data/images/

# Remove zip files
rm pretrain_data/images/images.zip
