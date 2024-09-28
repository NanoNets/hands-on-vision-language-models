# test.py
from torch_snippets.torch_loader import torch
from torch_snippets import P, PIL, Image, np


from vlm.base import VLM

class Llama_32_Base(VLM):
    def __init__(self, model_id):
        super().__init__()
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def predict(self, image, prompt, max_new_tokens=128):
        if isinstance(image, (P, str)):
            image = Image.open(image).convert('RGB')
        assert isinstance(image, PIL.Image.Image)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        res = self.processor.decode(output[0])
        return res

    @staticmethod
    def get_raw_output(pred):
        return pred


class Llama_32_11B(Llama_32_Base):
    def __init__(self):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        super().__init__(model_id)


class Llama_32_90B(Llama_32_Base):
    def __init__(self):
        model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
        super().__init__(model_id)