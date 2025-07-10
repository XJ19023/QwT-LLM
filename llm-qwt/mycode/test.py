
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

start_time = time.time()
# ----------------------------------------------------------
model_name = "/localssd/lbxj/Qwen2.5-0.5B"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print(type(model))
exit()

# prepare the model input
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# the result will begin with thinking content in <think></think> tags, followed by the actual response
print(tokenizer.decode(output_ids, skip_special_tokens=True))

# ----------------------------------------------------------
end_time = time.time()
duration = end_time - start_time
hour = duration // 3600
minute = (duration % 3600) // 60
second = duration % 60
print(f'>>> RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s\n')
