import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.build_model import MatMul
from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul
from copy import deepcopy


def quant_model_resnet(model, input_quant_params={}, weight_quant_params={}):

    # path_embed 8-bit
    embedding_input_quant_params = deepcopy(input_quant_params)
    embedding_input_quant_params['n_bits'] = 8
    embedding_weight_quant_params = deepcopy(weight_quant_params)
    embedding_weight_quant_params['n_bits'] = 8

    # head 8-bit
    head_input_quant_params = deepcopy(input_quant_params)
    head_input_quant_params['n_bits'] = 8
    head_weight_quant_params = deepcopy(weight_quant_params)
    head_weight_quant_params['n_bits'] = 8

    module_dict={}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        if isinstance(m, nn.Conv2d):
            idx = idx + 1 if idx != 0 else idx

            if 'conv1' == name:
                new_m = QuantConv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    m.dilation,
                    m.groups,
                    m.bias is not None,
                    embedding_input_quant_params,
                    embedding_weight_quant_params
                )
            else:
                new_m = QuantConv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    m.dilation,
                    m.groups,
                    m.bias is not None,
                    input_quant_params,
                    weight_quant_params
                )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            if 'fc' in name:
                new_m = QuantLinear(m.in_features, m.out_features, head_input_quant_params, head_weight_quant_params)
            else:
                raise NotImplementedError
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)

    return model


def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)
