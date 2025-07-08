import os
import random
import sys
import numpy as np
from safetensors.torch import load_file
import torch
import math
from safetensors.torch import save_file

import logging

import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 文件输出
file_handler = logging.FileHandler("log/data_processing.log", 'w')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
# 终端输出
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
# logger.addHandler(stream_handler)

import time
start_time = time.time()
# ----------------------------------------------------------

safetensors_path_act = '/cephfs/juxin/QwT/QwT-cls-RepQ-ViT/log/clamp.safetensors'
state_dict = load_file(safetensors_path_act)

org = state_dict['org']
clamp = state_dict['clamp']
clamp_final = state_dict['clamp_final']
clamp_idx = state_dict['clamp_idx']

# safetensors_path_act = '/cephfs/juxin/QwT/QwT-cls-RepQ-ViT/log/quant.safetensors'
# state_dict = load_file(safetensors_path_act)

# quant_final = state_dict['quant_final']


# print((~clamp_idx).sum())
max = org.amax(dim=-1, keepdim=True)
min = org.amin(dim=-1, keepdim=True)
present_range = max - min

print(present_range)





