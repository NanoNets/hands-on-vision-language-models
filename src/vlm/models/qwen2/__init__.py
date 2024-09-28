from torch_snippets import ifnone

from vlm.base import VLM


class Qwen2_base(VLM):
    def __init__(self, model, name=None, adapter=None):
        super().__init__()
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )
        if name is None and 'Custom' in self.__class__.__name__:
            raise ValueError("Custom Models need a name as argument for DB caching")
        self.name = ifnone(name, self.__class__.__name__)

        # default: Load the model on the available device(s)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model, torch_dtype="auto", device_map="auto"
        )
        if adapter is not None:
            self.model.load_adapter(adapter)

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def predict(self, image, prompt, max_new_tokens=1024):
        from qwen_vl_utils import process_vision_info

        img_b64_str, image_type = self.path_2_b64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:{image_type};base64,{img_b64_str}",
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    @staticmethod
    def get_raw_output(pred):
        return pred


class Qwen2_2B(Qwen2_base):
    def __init__(self):
        super().__init__("Qwen/Qwen2-VL-2B-Instruct")


class Qwen2_7B(Qwen2_base):
    def __init__(self):
        super().__init__("Qwen/Qwen2-VL-7B-Instruct")


class Qwen2_Custom(Qwen2_base): # /home/paperspace/Code/hands-on-vision-language-models/LLaMA-Factory/saves/cord/qwen2_vl-2b/lora/sft
    def __init__(self, model, name, adapter):
        super().__init__(model, name, adapter)
