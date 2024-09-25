import os, json
from torch_snippets import P, in_debug_mode

from vlm.base import VLM

class GPT4oMini(VLM):
    def __init__(self, token=None):
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI(api_key=token or os.environ.get('OPENAI_API_KEY'))

    def predict(self, image, prompt, *, image_size=None, **kwargs):
        img_b64_str, image_type = self.path_2_b64(image, image_size)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_type};base64,{img_b64_str}"},
                        },
                    ],
                }
            ],
        )
        # used to be response.choices[0].message.content
        return response.to_json()

    @staticmethod
    def get_raw_output(pred):
        pred = json.loads(pred)
        pred = pred['choices'][0]['message']['content']
        return pred


    
