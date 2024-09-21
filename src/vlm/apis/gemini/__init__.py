import os
import google.generativeai as genai
from torch_snippets import readPIL

class Gemini:
    def __init__(self, token=None):
        genai.configure(api_key=token or os.environ.get('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def __call__(self, image, prompt, **kwargs):
        image = readPIL(image)
        response = self.model.generate_content([prompt, image])
        return response.text