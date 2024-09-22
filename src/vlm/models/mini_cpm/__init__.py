# test.py
from PIL import Image
from torch_snippets.torch_loader import torch


from vlm.base import VLM

class MiniCPM(VLM):
    def __init__(self, _=None):
        from transformers import AutoModel, AutoTokenizer
        model_id = 'openbmb/MiniCPM-V-2_6'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=self.torch_dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def __call__(self, image, prompt):
        image = Image.open(image).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return res