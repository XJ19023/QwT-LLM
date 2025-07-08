import torch

n_samples = 10
seq_len = 2048
hidden_dim = 4096
output_tensor = torch.empty((n_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cuda")
for i in range(n_samples):
    hidden_states = torch.randn((1, seq_len, hidden_dim), dtype=torch.bfloat16, device="cuda")
    output_tensor[i] = hidden_states[0].detach()  # 只取 batch=0，或根据实际维度
    del hidden_states
    torch.cuda.empty_cache()


#=======================================================================
@torch.no_grad()
def cal_wandb_to_compensate(model, dataset, tokenizer, device, train_samples=None, clamp=None, model_name=None):
    dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids
    train_samples = train_samples if train_samples else dataset.size(1) // 2048
    model.eval()

    if 'opt' in args.model_name.lower():
        forward_before_blocks = model.model.decoder.forward_before_blocks
        layers = model.model.decoder.layers
    if 'llama' in args.model_name.lower():
        forward_before_blocks = model.model.forward_before_blocks
        layers = model.model.layers

    seq_len = 2048
    if model_name == 'opt-125m':
        hidden_dim = 768
    if model_name == 'opt-1.3b':
        hidden_dim = 2048
    if model_name == 'opt-2.7b':
        hidden_dim = 2560
    if model_name == 'opt-6.7b':
        hidden_dim = 4096
    if model_name == 'opt-13b':
        hidden_dim = 5120
    if model_name == 'TinyLlama-1.1B-Chat-v1.0':
        hidden_dim = 2048
    if model_name == 'llama-2-7b-hf':
        hidden_dim = 4096
    if model_name == 'Meta-Llama-3-8B':
        hidden_dim = 4096

    layer_inputs = torch.empty((train_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cuda")
    # layer_inputs = []
    for i in tqdm(range(train_samples), desc="Before layers..."):
        batch = dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            '''
            if i == 0:
                hidden_states, causal_attention_mask, use_cache = model.model.decoder.forward_before_blocks(batch)
            else:
                hidden_states, _, _ = forward_before_blocks(batch)
            '''
            if 'opt' in args.model_name.lower(): 
                hidden_states, causal_attention_mask, use_cache = forward_before_blocks(batch, return_dict=True)
            if 'llama' in args.model_name.lower(): 
                hidden_states, position_ids, past_key_values, use_cache, cache_position, position_embeddings = forward_before_blocks(batch)
            layer_inputs[i] = hidden_states[0].detach()
            # layer_inputs.append(hidden_states.detach())  # ⚠️ 挪回 CPU
            del batch, hidden_states
            torch.cuda.empty_cache()
    # # layer_inputs = torch.cat(layer_inputs, dim=0)
    # torch.save(layer_inputs, "./layer_inputs.pt")
    # torch.save(layer_inputs, "./layer_inputs.pt")
    # exit()
    # raise SystemExit

    for layer_idx, layer in tqdm(enumerate(layers), total=len(layers), desc="In layers"):
        layer_outputs = torch.empty((train_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cuda")
        layer_outputs_quant = torch.empty((train_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cuda")
        # layer_outputs = []
        # layer_outputs_quant = []
        for input_idx, layer_input in enumerate(layer_inputs):
            layer_input.unsqueeze_(0)
            '''
            if get_save_tensor_enable():
                append_activation(f'before_layers.{layer_idx}.input', layer_inputs)
                append_activation(f'before_layers.{layer_idx}.attention_mask', causal_attention_mask)
            with open('log/123.log', 'a') as f:
                f.writelines(f'before_layers.{layer_idx}.input.dtype, {layer_input.dtype}\t {layer_input.shape}\n')
            '''
            set_quant_state(layer, quant=False, clamp=False)
            if 'opt' in args.model_name.lower():
                layer_output = layer(layer_input, causal_attention_mask, use_cache=use_cache)
            if 'llama' in args.model_name.lower():
                layer_output = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            # layer_outputs.append(layer_output[0].detach())
            layer_outputs[input_idx] = layer_output[0][0].detach()
            set_quant_state(layer, quant=True, clamp=clamp)
            if 'opt' in args.model_name.lower():
                layer_output_quant = layer(layer_input, causal_attention_mask, use_cache=use_cache)
            if 'llama' in args.model_name.lower():
                layer_output_quant = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            # layer_outputs_quant.append(layer_output_quant[0].detach())
            layer_outputs_quant[input_idx] = layer_output_quant[0][0].detach()
        # # layer_outputs = torch.cat(layer_outputs, dim=0)
        # # layer_outputs_quant = torch.cat(layer_outputs_quant, dim=0)
        W, b, r2_score = lienar_regression(layer_inputs, layer_outputs - layer_outputs_quant, block_id=layer_idx)
        with open(f'log/{model_name}/r2_score.txt', 'a') as f:
            f.writelines(f'layer_idx: {layer_idx:>3}  r2_score: {r2_score}\n')
        # W, b, r2_score = torch.tensor([0]), torch.tensor([0]), 0
        if layer_idx > 5 and r2_score > 0:
            layer.set_qwt_para(W, b, r2_score)
            set_quant_state(layer, quant=True, clamp=clamp)
        else:
            layer.set_qwt_para(None, None, 0)
            set_quant_state(layer, quant=False, clamp=False)
        model.cuda()

        layer_outputs_wandb = torch.empty((train_samples, seq_len, hidden_dim), dtype=torch.bfloat16, device="cuda")
        # layer_outputs_wandb = []
        for input_idx, layer_input in enumerate(layer_inputs):
            layer_input.unsqueeze_(0)
            set_quant_state(layer, quant=True, clamp=clamp)
            if 'opt' in args.model_name.lower():
                layer_output_wandb = layer(layer_input, causal_attention_mask, use_cache=use_cache)
            if 'llama' in args.model_name.lower():
                layer_output_wandb = layer(layer_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            # layer_outputs_wandb.append(layer_output_wandb[0].detach())
            layer_outputs_wandb[input_idx] = layer_output_wandb[0][0].detach()
        # layer_inputs = torch.cat(layer_outputs_wandb, dim=0)
        layer_inputs = layer_outputs_wandb

    del layer_inputs, layer_outputs, layer_outputs_quant
    torch.cuda.empty_cache()
    # exit()
#=======================================================================