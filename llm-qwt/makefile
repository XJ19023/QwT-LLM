models = opt-125m TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Meta-Llama-3-8B Llama-2-13b-hf opt-125m opt-1.3b opt-2.7b opt-6.7b opt-13b
methods = --eval_base --eval_quant --eval_clamp --eval_quant_qwt --eval_clamp_qwt

models = TinyLlama-1.1B-Chat-v1.0 llama-2-7b-hf Meta-Llama-3-8B
methods = --eval_clamp_qwt

llama:
	TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=llama-2-7b-hf --n_samples=40 --eval_quant_qwt --eval_clamp_qwt
opt:
	TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=opt-2.7b --n_samples=40 --eval_clamp_qwt

vit:
	TOKENIZERS_PARALLELISM=false python mycode/vit.py


data:
	python mycode/data_process.py


save_tensor:
	./run.sh save_tensor llama-2-7b-hf W4A8
	
all_bk: 
	@for model in $(models); do \
		for method in $(methods); do \
			TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=$$model --$$method; \
		done; \
	done

all: 
	@for model in $(models); do \
		TOKENIZERS_PARALLELISM=false python mycode/llm.py --model_name=$$model --n_samples=1 $(methods); \
	done


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





