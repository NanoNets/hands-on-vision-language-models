from openai import OpenAI
from torch_snippets import P, in_debug_mode

from vlm.base import VLM

class GPT4oMini(VLM):
    def __init__(self, token):
        super().__init__(token)
        self.client = OpenAI(api_key=token)

    def __call__(self, image, prompt, *, image_size=None, **kwargs):
        img_b64_str = self.path_2_b64(image, image_size)
        image_type = f'image/{P(image).extn}'
        if in_debug_mode():
            print(image_type)
            return
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64_str}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
