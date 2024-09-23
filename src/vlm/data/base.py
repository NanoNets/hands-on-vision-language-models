import json, traceback, time
from torch_snippets import *
from vlm.available_vlms import VLMs
from vlm.base import VLM

def main(ds, vlm: str|VLM, prompt, n:int=None, image_key='images', dataset_name=None):
    if isinstance(vlm, str):
        assert vlm in VLMs, f'VLM {vlm} not found. Available VLMs: {list(VLMs.keys())}'
        vlm:VLM = VLMs[vlm]()

    _vlm = vlm.__class__.__name__

    n = len(ds) if n is None else n
    for ix,x in E(track2(ds)):
        if _vlm == 'Gemini': time.sleep(30)
        # Use vlm.cursor to fetch the actual predictions
        vlm(x[image_key], prompt=prompt, dataset_name=dataset_name, dataset_row_id=ix)
        if ix >= n:
            Info(f'Breaking Early as {n=} is set')
            break