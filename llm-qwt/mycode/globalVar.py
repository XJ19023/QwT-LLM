from safetensors.torch import save_file

iterationCounter = 0
def increas_iterationCounter():
    global iterationCounter
    iterationCounter += 1
    return iterationCounter
def get_iterationCounter():
    global iterationCounter
    return iterationCounter


activations_save = {}
weights_save = {}
def append_activation(k, v):
    global activations_save
    activations_save[f'{k}'] = v.detach().cpu()
def append_weight(k, v):
    global weights_save
    weights_save[f'{k}'] = v.detach().cpu()
def save_tensors(dir=None):
    global activations_save
    global weights_save
    activations_save = {k: v.clone().cpu() for k, v in activations_save.items()}
    with open(f'{dir}/activation_key.py', 'w') as f:
        f.writelines(f"act_keys = [")
        for k in activations_save.keys():
            f.writelines(f"'{k}',\n")
        f.writelines(f"]")
    weights_save = {k: v.clone().cpu() for k, v in weights_save.items()}
    with open(f'{dir}/weight_key.py', 'w') as f:
        f.writelines(f"wgt_keys = [")
        for k in weights_save.keys():
            f.writelines(f"'{k}',\n")
        f.writelines(f"]")
    save_file(activations_save, f"{dir}/activation.safetensors")
    save_file(weights_save, f"{dir}/weight.safetensors")

save_tensor_enable = False
def set_save_tensor_enable():
    global save_tensor_enable
    save_tensor_enable = True
def get_save_tensor_enable():
    global save_tensor_enable
    return save_tensor_enable

clamped_quant_enable = False
def set_clamped_quant_enable():
    global clamped_quant_enable
    clamped_quant_enable = True
def get_clamped_quant_enable():
    global clamped_quant_enable
    return clamped_quant_enable

profiling_enable = False
def set_profiling_enable():
    global profiling_enable
    profiling_enable = True
def get_profiling_enable():
    global profiling_enable
    return profiling_enable

data_type = None
def set_data_type(dataType):
    global data_type
    data_type = dataType
def get_data_type():
    global data_type
    return data_type