# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# from datasets import load_dataset
# tokenizer = AutoTokenizer.from_pretrained('/localssd/lbxj/Qwen2.5-0.5B')
# test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# dataset = tokenizer(
#         "\n\n".join(test_data["text"]), return_tensors="pt"
#     ).input_ids
# train_samples = dataset.size(1) // 2048

# print(train_samples)

# dataset = tokenizer(
#         "\n\n".join(train_data["text"]), return_tensors="pt"
#     ).input_ids
# train_samples = dataset.size(1) // 2048

# print(train_samples)
list = [1, 2, 4]
with open('log/123.log', 'w') as f:
    f.writelines(f'{list}')