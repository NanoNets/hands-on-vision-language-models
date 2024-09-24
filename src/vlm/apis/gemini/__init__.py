import os, time, json
from torch_snippets import readPIL, P, Image

from vlm.base import VLM

class Gemini(VLM):
    def __init__(self, token=None):
        super().__init__()
        import google.generativeai as genai
        genai.configure(api_key=token or os.environ.get('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def predict(self, image, prompt, **kwargs):
        if isinstance(image, (str, P)):
            image = readPIL(image)
        assert isinstance(image, Image.Image), f'Received image of type {type(image)}'
        response = self.model.generate_content([prompt, image])
        # used to be response.text
        return json.dumps(response.to_dict())

    @staticmethod
    def get_raw_output(pred):
        pred = json.loads(pred)
        pred = pred['candidates'][0]['content']['parts'][0]['text']
        return pred
