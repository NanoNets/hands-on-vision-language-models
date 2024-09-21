import os
import anthropic

from vlm.base import VLM

class Claude_35(VLM):
    def __init__(self, token=None):
        self.client = anthropic.Anthropic(api_key=token or os.environ['CLAUDE_API_KEY'])

    def __call__(self, image, prompt, max_tokens=1024, image_data=None):
        image_data, image_type = self.path_2_b64(image)
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=max_tokens,
            messages = [
                dict(role='user', content=[
                    dict(type='image', source=dict(type='base64', media_type=image_type, data=image_data)), 
                    dict(type='text', text=prompt)
                ])
            ]
        )
        return message