models = opt-125m TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Meta-Llama-3-8B Llama-2-13b-hf opt-125m opt-1.3b opt-2.7b opt-6.7b opt-13b Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-7B
models = TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Meta-Llama-3-8B Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-7B
methods = --eval_base --eval_quant --eval_clamp --eval_quant_qwt --eval_clamp_qwt

models = TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Meta-Llama-3-8B Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-7B
methods = --eval_clamp_qwt
tasks = wikitext c4
tasks = c4

qwen:
	TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=Qwen2.5-0.5B --n_samples=1 --save_tensor --eval_quant
llama:
	TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=Qwen3-8B --eval_clamp_qwt
tmp:
	TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=Qwen2.5-0.5B --eval_clamp_qwt --n_samples=1 --task=c4

opt:
	TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=opt-2.7b --n_samples=40 --eval_clamp_qwt

vit:
	TOKENIZERS_PARALLELISM=false python mycode/vit.py

data:
	python mycode/data_process.py


save_tensor:
	./run.sh save_tensor llama-2-7b-hf W4A8
	

all:
	@bash -c '\
	start=$$(date +%s); \
	for model in $(models); do \
		for task in $(tasks); do \
			echo "TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=$$model --task=$$task ${methods}"; \
			TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=$$model --task=$$task ${methods}; \
		done; \
	done; \
	end=$$(date +%s); \
	delta=$$((end - start)); \
	hours=$$((delta / 3600)); \
	minutes=$$(((delta % 3600) / 60)); \
	echo -e "\e[36mTime elapsed: $${hours}h-$${minutes}m\e[0m"; \
	'



test:
	python mycode/test.py
gpu:
	python shared_code/use_gpu.py

tar:
	./mycode/tar.sh

echo:
	@for model in $(models); do \
		echo $$model; \
		tail -n 1 log/$$model/no_quant.log; \
	done





