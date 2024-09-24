from torch_snippets.torch_loader import torch
import transformers, warnings
from PIL import Image

from vlm.base import VLM

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

class Bunny(VLM):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = 'cuda'  # or cpu
        torch.set_default_device(self.device)

        # create model
        self.model = AutoModelForCausalLM.from_pretrained(
            'BAAI/Bunny-v1_1-Llama-3-8B-V',
            torch_dtype=torch.float16, # float32 for cpu
            device_map=self.device,
            trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            'BAAI/Bunny-v1_1-Llama-3-8B-V',
            trust_remote_code=True)


    def predict(self, image, prompt):
        # text prompt
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(self.device)

        # image, sample images can be found in images folder
        image = Image.open(image)
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype, device=self.device)

        # generate
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=100,
            use_cache=True,
            repetition_penalty=1.0 # increase this to avoid chattering
        )[0]

        output_text = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        return output_text

    @staticmethod
    def get_raw_output(pred):
        return pred


