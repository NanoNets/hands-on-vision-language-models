import os
from torch_snippets import readPIL, P, Image

class Gemini:
    def __init__(self, token=None):
        import google.generativeai as genai
        genai.configure(api_key=token or os.environ.get('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def __call__(self, image, prompt, **kwargs):
        if isinstance(image, (str, P)):
            image = readPIL(image)
        assert isinstance(image, Image.Image), f'Received image of type {type(image)}'
        response = self.model.generate_content([prompt, image])
        return response.text