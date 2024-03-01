# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
# from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()

import torch
from hf_olmo import *

from llava.train.train import train

import faulthandler
# faulthandler.dump_traceback_later(20, repeat=True)

if __name__ == "__main__":
    train()
