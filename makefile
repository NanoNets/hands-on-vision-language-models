predict-qwen-cord:
	vlm predict-cord qwen2 --n=100

predict-qwen-sroie:
	vlm predict-sroie qwen2 --n=100

predict-gemini-cord:
	vlm predict-cord gemini --n=5

predict-gemini-sroie:
	vlm predict-sroie gemini --n=100

predict-gpt4omini-cord:
	vlm predict-cord gpt4omini --n=5

predict-gpt4omini-sroie:
	vlm predict-sroie gpt4omini --n=100

all:
	make predict-qwen-cord
	make predict-qwen-sroie
	make predict-gpt4omini-cord
	make predict-gpt4omini-sroie
	make predict-gemini-cord
	make predict-gemini-sroie