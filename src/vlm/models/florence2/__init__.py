from transformers import AutoProcessor, AutoModelForCausalLM
from torch_snippets import readPIL, AD
from torch_snippets.torch_loader import torch

from vlm.base import VLM

class Florence2(VLM):
    def __init__(self, _=None):
        model_id = 'microsoft/Florence-2-large'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device).half()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def __call__(self, image, task):
        image = readPIL(image)
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
        return AD(parsed_answer)