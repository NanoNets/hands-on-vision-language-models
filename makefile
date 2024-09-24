N ?= 100

predict-Qwen2-cord:
	vlm predict-cord Qwen2 --n=$(N)

predict-Qwen2-sroie:
	vlm predict-sroie Qwen2 --n=$(N)

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
	make predict-Claude_35-cord
	make predict-Claude_35-sroie
	make predict-Qwen2-cord
	make predict-Qwen2-sroie
	make predict-GPT4oMini-cord
	make predict-GPT4oMini-sroie
	make predict-Gemini-cord
	make predict-Gemini-sroie
