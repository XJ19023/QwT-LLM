'''
    -计算int4 int5_6 int7_8的占比
'''
import os
import random
import sys
import numpy as np
from safetensors.torch import load_file
import torch
import math
from safetensors.torch import save_file
from matplotlib import pyplot as plt

import time
start_time = time.time()
# ----------------------------------------------------------
methods = ['qwt_wikitext', 'qwt_c4'] # qwt is wikitext

models = [
            'TinyLlama-1.1B-Chat-v1.0',
            'llama-2-7b-hf',
            'Meta-Llama-3-8B',
            'Qwen2.5-0.5B',
            'Qwen2.5-1.5B',
            'Qwen2.5-7B'
            ]
# models = ['Qwen2.5-0.5B']
for method in methods:
    for model in models:
        print(f'======={model}=======')
        if model == 'TinyLlama-1.1B-Chat-v1.0':
            from saved_tensor.TinyLlama_1_1B_Chat_v1_0_qwt_c4.activation_key import act_keys
        elif model == 'llama-2-7b-hf':
            from saved_tensor.llama_2_7b_hf_qwt_c4.activation_key import act_keys
        elif model == 'Meta-Llama-3-8B':
            from saved_tensor.Meta_Llama_3_8B_qwt_c4.activation_key import act_keys
        elif model == 'Qwen2.5-0.5B':
            from saved_tensor.Qwen2_5_0_5B_qwt_c4.activation_key import act_keys
        elif model == 'Qwen2.5-1.5B':
            from saved_tensor.Qwen2_5_1_5B_qwt_c4.activation_key import act_keys
        elif model == 'Qwen2.5-7B':
            from saved_tensor.Qwen2_5_7B_qwt_c4.activation_key import act_keys

        model_rename = model.replace("-", "_").replace(".", "_")
        safetensors_path_act = f"saved_tensor/{model_rename}_{method}/activation.safetensors"
        state_dict = load_file(safetensors_path_act)
        keys = act_keys

        profile_data = True
        if profile_data:
            int4_counter_org = int5_6_counter_org = int7_8_counter_org = 0
            int4_counter_shift = int5_6_counter_shift = int7_8_counter_shift = 0
            sub_tensor_count = 0    
            with open(f'intRatio_{method}.txt', 'a') as f:
                f.writelines(f'--{model_rename}\n')
                for idx, key in enumerate(keys):
                    w_int8 = state_dict[key].to('cuda').squeeze()
                    w_int8 = w_int8.reshape(-1, 128)
                    max = w_int8.amax(dim=-1, keepdim=True)
                    min = w_int8.amin(dim=-1, keepdim=True)
                    present_range = max
                    int4 = present_range <= 15
                    int4_counter_org += int4.sum()
                    int5_6 = (present_range <= 63) * (present_range > 15)
                    int5_6_counter_org += int5_6.sum()
                    int7_8 = present_range > 63
                    int7_8_counter_org += int7_8.sum()

                    present_range = max - min
                    int4 = present_range <= 15
                    int4_counter_shift += int4.sum()
                    int5_6 = (present_range <= 63) * (present_range > 15)
                    int5_6_counter_shift += int5_6.sum()
                    int7_8 = present_range > 63
                    int7_8_counter_shift += int7_8.sum()
                    sub_tensor_count += w_int8.size(0)
            
                assert (int4_counter_org + int5_6_counter_org + int7_8_counter_org) == sub_tensor_count
                f.writelines(f'int4: {int4_counter_org:>8} ({int4_counter_org*100/sub_tensor_count:5.2f}%), int5_6: {int5_6_counter_org:>8} ({int5_6_counter_org*100/sub_tensor_count:5.2f}%), int7_8: {int7_8_counter_org:>8} ({int7_8_counter_org*100/sub_tensor_count:5.2f}%)\n')
                f.writelines(f'int4: {int4_counter_shift:>8} ({int4_counter_shift*100/sub_tensor_count:5.2f}%), int5_6: {int5_6_counter_shift:>8} ({int5_6_counter_shift*100/sub_tensor_count:5.2f}%), int7_8: {int7_8_counter_shift:>8} ({int7_8_counter_shift*100/sub_tensor_count:5.2f}%)\n\n')
    # ----------------------------------------------------------
    with open(f'intRatio_{method}.txt', 'a') as f:
        end_time = time.time()
        duration = end_time - start_time
        hour = duration // 3600
        minute = (duration % 3600) // 60
        second = duration % 60
        f.writelines(f'>>>RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s\n\n')
                                




