'''
to speedup debug, set length=512, n_sample=1, use opt-125m
开源代码量化后的结果向中间结果做线性回归，此版本修改为向全精度对齐
'''
import os
import random
import shutil
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import torch.distributed as dist

from datasets import load_dataset
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
from datetime import datetime
import sys
from quant import quantLinear
from globalVar import (increas_iterationCounter,
                       save_tensors,
                       set_save_tensor_enable,
                       set_data_type,
                       append_activation,
                       set_profiling_enable)

def gather_tensor_from_multi_processes(input, world_size):
    if world_size == 1:
        return input
    torch.cuda.synchronize()
    dist.all_gather(gathered_tensors, input)
    gathered_tensors = torch.cat(gathered_tensors, dim=0)
    torch.cuda.synchronize()

    return gathered_tensors
def lienar_regression(X, Y, block_id=0):
    X = X.to('cuda')
    Y = Y.to('cuda')
    X = X.reshape(-1, X.size(-1)).float()

    X = gather_tensor_from_multi_processes(X, world_size=args.world_size)

    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1)).float()

    Y = gather_tensor_from_multi_processes(Y, world_size=args.world_size)

    # _write('the shape of X_add_one is {}, Y is {}'.format(X_add_one.size(), Y.size()))

    X_add_one_T = X_add_one.t()
    W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y

    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    Y_pred = X @ W + b

    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot
    # _write('block : {}      abs : {:.6f}      r2 : {:.3f}'.format(block_id, abs_loss, r2_score))
    del X, Y
    torch.cuda.empty_cache()
    return W, b, r2_score
def tensor_gpu_memory(tensor, name=None):
    if tensor.is_cuda:
        size_in_bytes = tensor.element_size() * tensor.numel()
        size_in_MB = size_in_bytes / 1024**2
        print(f"{name if name else ''} size: {size_in_MB:.2f} MB")
    else:
        print(f"{name if name else ''} is not on GPU.")
# current date and time
current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
start_time = time.time()
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_name", type=str, default="opt-125m")
parser.add_argument("--task", type=str, default="wikitext")
parser.add_argument(
    "--act_scales_path",
    type=str,
    default="act_scales/llama-2-7b.pt",
)
parser.add_argument("--n_samples", type=int, default=None)
# parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument('--start_block', default=0, type=int)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--wgt_nbit", default=4, type=int)
parser.add_argument("--act_nbit", default=8, type=int)
parser.add_argument("--eval_base", action="store_true")
parser.add_argument("--eval_quant", action="store_true")
parser.add_argument("--eval_clamp", action="store_true")
parser.add_argument("--eval_quant_qwt", action="store_true")
parser.add_argument("--eval_clamp_qwt", action="store_true")
parser.add_argument("--profiling", action="store_true")
parser.add_argument("--save_tensor", action="store_true")
args = parser.parse_args()

@torch.no_grad()
def evaluate(model, dataset, n_samples=None):
    model.eval()
    nlls = []
    length = 2048
    n_samples = n_samples if n_samples else dataset.size(1) // length
    for i in tqdm(range(n_samples), desc="Evaluating..."):
        batch = dataset[:, (i * length) : ((i + 1) * length)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = dataset[:, (i * length) : ((i + 1) * length)][:, 1:].to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * length
        nlls.append(neg_log_likelihood)

        _ = increas_iterationCounter()

    return torch.exp(torch.stack(nlls).sum() / (n_samples * length))

args.world_size = 1
args.rank = 0  # global rank
alpha = args.alpha
model_path = '/localssd/lbxj/' + args.model_name
# model_path = '/cephfs/juxin/models/' + args.model_name
act_scales_path = args.act_scales_path
n_samples = args.n_samples
train_samples = 64 # 64
# set_save_tensor_enable()


tokenizer = AutoTokenizer.from_pretrained(model_path)
# test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# evaluator = Evaluator(test_data, tokenizer, "cuda", n_samples=n_samples)

def get_wikitext2(tokenizer, eval=True):
    if eval:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids
    return testenc

def get_c4(seqlen, tokenizer, eval=False):
    if eval:
        valdata = load_dataset(
            "json",
            data_files={"validation": "/cephfs/shared/juxin/dataset/c4/en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    else:
        valdata = load_dataset(
            "json",
            data_files={"train": "/cephfs/shared/juxin/dataset/c4/en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        if tmp.input_ids.shape[1] == seqlen:
            # rare case, discovered with Yi tokenizer
            valenc.append(tmp.input_ids)
        else:
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    return valenc    


if args.task == 'wikitext':
    test_data = get_wikitext2(tokenizer, True)
    train_data = get_wikitext2(tokenizer, False)
if args.task == 'c4':
    test_data = get_c4(2048, tokenizer, True)
    train_data = get_c4(2048, tokenizer, True)

os.makedirs(f'log/{args.model_name}', exist_ok=True)
with open('log/Running.log', 'a') as f:
    f.writelines(f'Running {args.model_name} at {current_time}\n')

# '''
import importlib.util
# replace modeling_bert.py
# 1. 加载你本地的 modeling_bert.py 文件
if 'llama' in args.model_name.lower():
    spec = importlib.util.spec_from_file_location(
        "transformers.models.llama.modeling_llama", 
        "./mycode/modeling_llama.py"
    )
    custom_llama = importlib.util.module_from_spec(spec)
    sys.modules["transformers.models.llama.modeling_llama"] = custom_llama
    spec.loader.exec_module(custom_llama)
if 'opt' in args.model_name.lower():
    spec = importlib.util.spec_from_file_location(
        "transformers.models.opt.modeling_opt", 
        "./mycode/modeling_opt.py"
    )
    custom_opt = importlib.util.module_from_spec(spec)
    sys.modules["transformers.models.opt.modeling_opt"] = custom_opt
    spec.loader.exec_module(custom_opt)
if 'qwen3' in args.model_name.lower():
    spec = importlib.util.spec_from_file_location(
        "transformers.models.qwen3.modeling_qwen3", 
        "./mycode/modeling_qwen3.py"
    )
    custom_opt = importlib.util.module_from_spec(spec)
    sys.modules["transformers.models.qwen3.modeling_qwen3"] = custom_opt
    spec.loader.exec_module(custom_opt)
if 'qwen2' in args.model_name.lower():
    spec = importlib.util.spec_from_file_location(
        "transformers.models.qwen2.modeling_qwen2", 
        "./mycode/modeling_qwen2.py"
    )
    custom_opt = importlib.util.module_from_spec(spec)
    sys.modules["transformers.models.qwen2.modeling_qwen2"] = custom_opt
    spec.loader.exec_module(custom_opt)
def set_quant_state(model, quant=True, clamp=False):
    for name, module in model.named_modules():
        if isinstance(module, quantLinear):
            module.enable_quant(quant, clamp)
# 2. 然后再加载模型
# '''
config = AutoConfig.from_pretrained(model_path)
config.use_cache = False  # ✅ 显式修改
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    config=config
)

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and name != 'lm_head':
        new_layer = quantLinear.set_param(module, name=name, wgt_nbit=args.wgt_nbit, act_nbit=args.act_nbit)
        _set_module(model, name, new_layer)

#=======================================================================
@torch.no_grad()
def cal_wandb_to_full(model, task, dataset, train_samples=None, clamp=None, model_name=None):
    train_samples = train_samples if train_samples else dataset.size(1) // 2048
    model.eval()

    if 'opt' in args.model_name.lower():
        forward_before_blocks = model.model.decoder.forward_before_blocks
        layers = model.model.decoder.layers
    if 'llama' in args.model_name.lower():
        forward_before_blocks = model.model.forward_before_blocks
        layers = model.model.layers
    if 'qwen' in args.model_name.lower():
        forward_before_blocks = model.model.forward_before_blocks
        layers = model.model.layers
    seq_len = 2048

    mx_accept_mse = 10000
    qwt_begin_block = -1
    layer_quant_qwt = [10000] # quant no_clamp qwt
    except_layer = [10000] # quant no clamp no qwt

    if task == 'wikitext':
        if model_name == 'TinyLlama-1.1B-Chat-v1.0':
            hidden_dim = 2048
            train_samples = 256
            except_layer = [0, 1, 2]
            layer_quant_qwt = [20]
        if model_name == 'llama-2-7b-hf':
            hidden_dim = 4096
            train_samples = 128
            except_layer = [0, 1, 2, 3]
            layer_quant_qwt = [31]
        if model_name == 'Llama-2-13b-hf':
            hidden_dim = 5120
            train_samples = 128
            layer_quant_qwt = [25, 33, 36, 37, 39]
            qwt_begin_block = 3
        if model_name == 'llama-13b':
            hidden_dim = 5120
            train_samples = 128
            qwt_begin_block = 3
            layer_quant_qwt = [7, 19, 33, 37]
            # qwt_begin_block = 7
        if model_name == 'Meta-Llama-3-8B':
            hidden_dim = 4096
            train_samples = 128
            except_layer = [0, 1, 2]
            layer_quant_qwt = [27, 29, 30, 31]

        if model_name == 'Qwen2.5-0.5B':
            hidden_dim = 896
            train_samples = 256
            layer_quant_qwt = [2, 22, 23]
            layer_quant_qwt = [2, 14, 22, 23]
        if model_name == 'Qwen2.5-1.5B':
            hidden_dim = 1536
            train_samples = 256
            except_layer = [0, 1, 2]
            layer_quant_qwt = [20, 23, 25, 27]
        if model_name == 'Qwen2.5-7B':
            hidden_dim = 3584
            train_samples = 128
            layer_quant_qwt = [23, 24, 25, 26, 27]
        if model_name == 'Qwen2.5-14B':
            hidden_dim = 5120
            train_samples = 128
            # layer_quant_qwt = [41, 42, 43, 44]
            # qwt_begin_block = 2
    if task == 'c4':
        if model_name == 'TinyLlama-1.1B-Chat-v1.0':
            hidden_dim = 2048
            train_samples = 256
            except_layer = [0, 1, 2]
            layer_quant_qwt = [20, 21]
        if model_name == 'llama-2-7b-hf':
            hidden_dim = 4096
            train_samples = 128
            except_layer = [0, 1, 2, 3]
            layer_quant_qwt = [31]
        if model_name == 'Llama-2-13b-hf':
            hidden_dim = 5120
            train_samples = 128
            except_layer = [0, 1, 2, 3]
            layer_quant_qwt = [25, 33, 34, 35, 39]
        if model_name == 'llama-13b':
            hidden_dim = 5120
            train_samples = 128
            qwt_begin_block = 3
            # except_layer = [0, 1, 2, 3]
            layer_quant_qwt = [6, 18, 32, 36]
        if model_name == 'Meta-Llama-3-8B':
            hidden_dim = 4096
            train_samples = 128
            except_layer = [0, 1, 2]
            layer_quant_qwt = [23, 28, 29, 30, 31]

        if model_name == 'Qwen2.5-0.5B':
            hidden_dim = 896
            train_samples = 256
            layer_quant_qwt = [2, 22, 23]
            layer_quant_qwt = [2, 14, 22, 23]
        if model_name == 'Qwen2.5-1.5B':
            hidden_dim = 1536
            train_samples = 256
            except_layer = [0, 1, 2]
            layer_quant_qwt = [20, 23, 24, 25, 27]
        if model_name == 'Qwen2.5-7B':
            hidden_dim = 3584
            train_samples = 128
            layer_quant_qwt = [23, 24, 25, 26, 27]

    # train_samples = 16
        
    with open(f'log/{model_name}/r2_score_{task}.txt', 'a') as f:
        f.writelines(f'lyr_qt_qwt={layer_quant_qwt}, ect_lyr={except_layer}\n')
    layer_inputs = torch.empty((train_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cpu")
    for i in tqdm(range(train_samples), desc="Before layers..."):
        batch = dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            if 'opt' in args.model_name.lower():
                hidden_states, causal_attention_mask, use_cache = forward_before_blocks(batch)
            if 'llama' in args.model_name.lower():
                hidden_states, position_ids, past_key_values, use_cache, cache_position, position_embeddings = forward_before_blocks(batch)
            if 'qwen' in args.model_name.lower():
                hidden_states, position_ids, cache_position, position_embeddings = forward_before_blocks(batch)
            layer_inputs[i] = hidden_states[0].detach().cpu()
            del batch, hidden_states
            torch.cuda.empty_cache()

    layer_inputs_full = layer_inputs
    for layer_idx, layer in tqdm(enumerate(layers), total=len(layers), desc="In layers"):
        layer_outputs = torch.empty((train_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cpu")
        layer_outputs_quant = torch.empty((train_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cpu")
        for input_idx, (layer_input, layer_input_full) in enumerate(zip(layer_inputs, layer_inputs_full)):
            layer_input = layer_input.to('cuda')
            layer_input_full = layer_input_full.to('cuda')
            layer_input.unsqueeze_(0)
            layer_input_full.unsqueeze_(0)
            '''
            if get_save_tensor_enable():
                append_activation(f'before_layers.{layer_idx}.input', layer_inputs)
                append_activation(f'before_layers.{layer_idx}.attention_mask', causal_attention_mask)
            with open('log/123.log', 'a') as f:
                f.writelines(f'before_layers.{layer_idx}.input.dtype, {layer_input.dtype}\t {layer_input.shape}\n')
            '''
            set_quant_state(layer, quant=False, clamp=False)
            # append_activation(f'qwt_layers.{layer_idx}.input_full', layer_input_full)
            if 'opt' in args.model_name.lower():
                layer_output = layer(layer_input_full, causal_attention_mask, use_cache=use_cache)
            if 'llama' in args.model_name.lower():
                ''' For debug
                with open('log/123_qwt.log', 'a') as f:
                    f.writelines(f'>>>> QwT {input_idx} <<<<\n')
                    f.writelines(f'input_ids type: {type(layer_input_full)} {layer_input_full.shape}\n')
                    # f.writelines(f'attention_mask type: {type(attention_mask)} {attention_mask}\n')
                    f.writelines(f'position_ids type: {type(position_ids)} {position_ids}\n')
                    f.writelines(f'past_key_values type: {type(past_key_values)} {past_key_values}\n')
                    # f.writelines(f'inputs_embeds type: {type(inputs_embeds)} {inputs_embeds}\n')
                    f.writelines(f'use_cache type: {type(use_cache)} {use_cache}\n')
                    # f.writelines(f'output_attentions type: {type(output_attentions)} {output_attentions}\n')
                    # f.writelines(f'output_hidden_states type: {type(output_hidden_states)} {output_hidden_states}\n')
                    f.writelines(f'cache_position type: {type(cache_position)} {cache_position}\n\n')
                    f.writelines(f'position_embeddings type: {type(position_embeddings)}\n\n')
                '''
                layer_output = layer(layer_input_full, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            if 'qwen' in args.model_name.lower():
                layer_output = layer(layer_input_full, position_ids=position_ids, cache_position=cache_position, position_embeddings=position_embeddings)
            layer_outputs[input_idx] = layer_output[0][0].detach().cpu()
            set_quant_state(layer, quant=True, clamp=clamp)
            if layer_idx in layer_quant_qwt:
                set_quant_state(layer, quant=True, clamp=False)
            if 'opt' in args.model_name.lower():
                layer_output_quant = layer(layer_input, causal_attention_mask, use_cache=use_cache)
            if 'llama' in args.model_name.lower():
                layer_output_quant = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            if 'qwen' in args.model_name.lower():
                layer_output_quant = layer(layer_input, position_ids=position_ids, cache_position=cache_position, position_embeddings=position_embeddings)
            layer_outputs_quant[input_idx] = layer_output_quant[0][0].detach().cpu()
        loss = nn.MSELoss()
        quant_loss = loss(layer_outputs.float(), layer_outputs_quant.float())
        W, b, r2_score = lienar_regression(layer_inputs, layer_outputs - layer_outputs_quant, block_id=layer_idx)
        if layer_idx > qwt_begin_block and r2_score > 0 and layer_idx not in except_layer:
            layer.set_qwt_para(W, b, r2_score)
            set_quant_state(layer, quant=True, clamp=clamp)
            if layer_idx in layer_quant_qwt:
                set_quant_state(layer, quant=True, clamp=False)
            for input_idx, layer_input in enumerate(layer_inputs):
                layer_input = layer_input.to('cuda')
                layer_input.unsqueeze_(0)
                if 'opt' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, causal_attention_mask, use_cache=use_cache)
                if 'llama' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
                if 'qwen' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, position_ids=position_ids, cache_position=cache_position, position_embeddings=position_embeddings)
                layer_outputs_quant[input_idx] = layer_output_wandb[0][0].detach().cpu()
            qwt_loss = loss(layer_outputs_quant.float(), layer_outputs.float())
            with open(f'log/{model_name}/r2_score_{task}.txt', 'a') as f:
                f.writelines(f'{layer_idx} r2_score: {r2_score:>14.5f}, quant_loss: {quant_loss:>12.5f}, qwt_mse: {qwt_loss:>12.5f}\n')
            # if qwt_loss > mx_accept_mse:
            if qwt_loss > quant_loss:
                layer.set_qwt_para(None, None, 0)
                set_quant_state(layer, quant=True, clamp=False)
                for input_idx, layer_input in enumerate(layer_inputs):
                    layer_input = layer_input.to('cuda')
                    layer_input.unsqueeze_(0)
                    if 'opt' in args.model_name.lower():
                        layer_output_wandb = layer(layer_input, causal_attention_mask, use_cache=use_cache)
                    if 'llama' in args.model_name.lower():
                        layer_output_wandb = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
                    if 'qwen' in args.model_name.lower():
                        layer_output_wandb = layer(layer_input, position_ids=position_ids, cache_position=cache_position, position_embeddings=position_embeddings)
                    layer_outputs_quant[input_idx] = layer_output_wandb[0][0].detach().cpu()
            layer_inputs = layer_outputs_quant
        else:
            with open(f'log/{model_name}/r2_score_{task}.txt', 'a') as f:
                f.writelines(f'{layer_idx} r2_score: {r2_score:>14.5f}, quant_loss: {quant_loss:>12.5f}\n')           
            layer.set_qwt_para(None, None, 0)
            set_quant_state(layer, quant=True, clamp=False)
            for input_idx, layer_input in enumerate(layer_inputs):
                layer_input = layer_input.to('cuda')
                layer_input.unsqueeze_(0)
                if 'opt' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, causal_attention_mask, use_cache=use_cache)
                if 'llama' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
                if 'qwen' in args.model_name.lower():
                    layer_output_wandb = layer(layer_input, position_ids=position_ids, cache_position=cache_position, position_embeddings=position_embeddings)
                layer_outputs_quant[input_idx] = layer_output_wandb[0][0].detach().cpu()
            layer_inputs = layer_outputs_quant
        layer_inputs_full = layer_outputs
        model.cuda()

        

    del layer_inputs, layer_inputs_full, layer_outputs, layer_outputs_quant
    torch.cuda.empty_cache()

    return train_samples, layer_quant_qwt, except_layer
    # exit()
#=======================================================================

if args.eval_base:
    print(f'---eval {args.model_name} base---')
    if args.save_tensor:
        set_save_tensor_enable()
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'base {args.model_name} PPL: {ppl}')
        saved_name = args.model_name.replace("-", "_").replace(".", "_")
        dir=f'/cephfs/shared/juxin/saved_tensor/qwt/{saved_name}_base'
        os.makedirs(dir, exist_ok=True)
        save_tensors(dir=dir)
    else:
        with open(f'log/{args.model_name}/structure_base.txt', 'w') as f:
            f.writelines(f'{type(model)}\n\n{model}')
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'base {args.model_name} PPL: {ppl}')
        with open(f'log/{args.model_name}/ppl.txt', 'a') as f:
            f.writelines(f'base PPL: {ppl}\n')
if args.eval_quant:
    print(f'---eval {args.model_name} quant---')
    set_quant_state(model, quant=True, clamp=False)
    if args.save_tensor:
        set_save_tensor_enable()
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'W{args.wgt_nbit}A{args.act_nbit} {args.model_name} PPL: {ppl}')
        saved_name = args.model_name.replace("-", "_").replace(".", "_")
        dir=f'/cephfs/shared/juxin/saved_tensor/qwt/{saved_name}_quant'
        os.makedirs(dir, exist_ok=True)
        save_tensors(dir=dir)
    else:
        with open(f'log/{args.model_name}/structure_quant.txt', 'w') as f:
            f.writelines(f'{type(model)}\n\n{model}')
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'W{args.wgt_nbit}A{args.act_nbit} {args.model_name} PPL: {ppl}')
        with open(f'log/{args.model_name}/ppl.txt', 'a') as f:
            f.writelines(f'W{args.wgt_nbit}A{args.act_nbit} PPL: {ppl}\n')
if args.eval_clamp:
    print(f'---eval clamp---')
    set_quant_state(model, quant=True, clamp=True)
    with open(f'log/{args.model_name}/structure_clamp.txt', 'w') as f:
        f.writelines(f'\n{type(model)}\n\n{model}')
    ppl = evaluate(model, test_data, args.n_samples)
    print(f'clamp PPL: {ppl}')
    with open(f'log/{args.model_name}/ppl.txt', 'a') as f:
        f.writelines(f'clamp PPL: {ppl}\n')
if args.eval_quant_qwt:
    print(f'---eval {args.model_name} quant qwt---')
    if args.save_tensor:
        _, _, _ = cal_wandb_to_full(model, args.task, train_data, train_samples, clamp=False, model_name=args.model_name)
        set_save_tensor_enable()
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'quant {args.model_name} PPL: {ppl}')
        saved_name = args.model_name.replace("-", "_").replace(".", "_")
        dir=f'/cephfs/shared/juxin/saved_tensor/qwt/{saved_name}_qwt_{args.task}'
        os.makedirs(dir, exist_ok=True)
        save_tensors(dir=dir)
    else:
        with open(f'log/{args.model_name}/r2_score_{args.task}.txt', 'a') as f:
            f.writelines(f'\n==========train quant qwt====={args.task}======= ') 
        train_samples, layer_quant_qwt, except_layer = cal_wandb_to_full(model, args.task, train_data, train_samples, clamp=False, model_name=args.model_name)
        with open(f'log/{args.model_name}/structure_quant_qwt_{args.task}.txt', 'w') as f:
            f.writelines(f'{type(model)}\n\n{model}')
        if args.profiling:
            set_profiling_enable()
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'quant qwt PPL: {ppl}')
        with open(f'log/{args.model_name}/ppl_{args.task}.txt', 'a') as f:
            f.writelines(f'qnt qwt PPL: {ppl}, trn_spls={train_samples}, lyr_qt_qwt={layer_quant_qwt}, ect_lyr={except_layer}\n')

    if args.profiling:
        dst = f'log/{args.model_name}/profiling.txt'
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move('log/profiling.txt', dst)
if args.eval_clamp_qwt:
    print(f'---eval {args.model_name} clamp qwt---')
    if args.save_tensor:
        _, _, _ = cal_wandb_to_full(model, args.task, train_data, train_samples, clamp=True, model_name=args.model_name)
        set_save_tensor_enable()
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'quant {args.model_name} PPL: {ppl}')
        saved_name = args.model_name.replace("-", "_").replace(".", "_")
        dir=f'/cephfs/shared/juxin/saved_tensor/qwt/{saved_name}_qwt_{args.task}'
        os.makedirs(dir, exist_ok=True)
        save_tensors(dir=dir)
    else:
        with open(f'log/{args.model_name}/r2_score_{args.task}.txt', 'a') as f:
            f.writelines(f'\n==========train clamp qwt====={args.task}======= ') 
        train_samples, layer_quant_qwt, except_layer = cal_wandb_to_full(model, args.task, train_data, train_samples, clamp=True, model_name=args.model_name)
        with open(f'log/{args.model_name}/structure_clamp_qwt_{args.task}.txt', 'w') as f:
            f.writelines(f'{type(model)}\n\n{model}')
        if args.profiling:
            set_profiling_enable()
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'clamp qwt PPL: {ppl}')
        with open(f'log/{args.model_name}/ppl_{args.task}.txt', 'a') as f:
            f.writelines(f'clp qwt PPL: {ppl}, trn_spls={train_samples}, lyr_qt_qwt={layer_quant_qwt}, ect_lyr={except_layer}\n')

    if args.profiling:
        dst = f'log/{args.model_name}/profiling.txt'
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move('log/profiling.txt', dst)

# ----------------------------------------------------------
end_time = time.time()
duration = end_time - start_time
hour = duration // 3600
minute = (duration % 3600) // 60
second = duration % 60
print(f'>>> RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s\n')
with open(f'log/{args.model_name}/ppl.txt', 'a') as f:
    f.writelines(f'>>> RUNNING TIME {args.task}: {int(hour)}h-{int(minute)}m-{int(second)}s  {current_time}\n\n')


