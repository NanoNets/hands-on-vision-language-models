N ?= 100

predict-Qwen2-cord:
	vlm predict-cord Qwen2 --n=$(N)
	
predict-Qwen2-custom-cord:
	# vlm predict-cord Qwen2_Custom --n=$(N) --vlm-kwargs='{"model": "Qwen/Qwen2-VL-2B-Instruct", "name": "Qwen-2B-Cord-Finetuned", "adapter": "/home/paperspace/LLaMA-Factory/saves/cord/qwen2_vl-2b/lora/sft"}'
	# vlm predict-cord Qwen2_Custom --n=$(N) --vlm-kwargs='{"model": "Qwen/Qwen2-VL-2B-Instruct", "name": "Qwen-2B-Cord-v3-Finetuned", "adapter": "/home/paperspace/LLaMA-Factory/saves/cord-3/qwen2_vl-2b/lora/sft"}'
	vlm predict-cord Qwen2_Custom --n=$(N) --vlm-kwargs='{"model": "Qwen/Qwen2-VL-2B-Instruct", "name": "Qwen-2B-Cord-v4-Finetuned", "adapter": "/home/paperspace/LLaMA-Factory/saves/cord-4/qwen2_vl-2b/lora/sft"}'

predict-Qwen2-sroie:
	vlm predict-sroie Qwen2 --n=$(N)

predict-MiniCPM-cord:
	vlm predict-cord MiniCPM --n=$(N)

predict-MiniCPM-sroie:
	vlm predict-sroie MiniCPM --n=$(N)

predict-Bunny-cord:
	vlm predict-cord Bunny --n=$(N)

predict-Bunny-sroie:
	vlm predict-sroie Bunny --n=$(N)

predict-Gemini-cord:
	vlm predict-cord Gemini --n=$(N)

predict-Gemini-sroie:
	vlm predict-sroie Gemini --n=$(N)

predict-GPT4oMini-cord:
	vlm predict-cord GPT4oMini --n=$(N)

predict-GPT4oMini-sroie:
	vlm predict-sroie GPT4oMini --n=$(N)

predict-Claude_35-cord:
	vlm predict-cord Claude_35 --n=$(N)

predict-Claude_35-sroie:
	vlm predict-sroie Claude_35 --n=$(N)

all:
	make predict-Bunny-cord
	make predict-Bunny-sroie

all-old:
	make predict-Claude_35-cord
	make predict-Claude_35-sroie
	make predict-Qwen2-cord
	make predict-Qwen2-sroie
	make predict-GPT4oMini-cord
	make predict-GPT4oMini-sroie
	make predict-Gemini-cord
	make predict-Gemini-sroie
	make predict-MiniCPM-cord
	make predict-MiniCPM-sroie