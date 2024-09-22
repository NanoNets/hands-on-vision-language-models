import json, traceback
from torch_snippets import *
from vlm.available_vlms import VLMs

def main(ds, vlm, prompt, db_folder, n:int=None):
    from google.api_core.exceptions import ResourceExhausted
    if isinstance(vlm, str):
        assert vlm in VLMs, f'VLM {vlm} not found. Available VLMs: {list(VLMs.keys())}'
        vlm = VLMs[vlm]()

    n = len(ds) if n is None else n
    for ix,x in E(track2(ds)):
        _to = db_folder/f'{ix}.json'
        if exists(_to): continue
        makedir(parent(_to))
        x = AD(x)
        try:
            pred = vlm(x.images, prompt=prompt).lstrip('```json').rstrip('```')
        except ResourceExhausted as e:
            Warn(f'Resource Exhausted {e} @ {ix}::{traceback.format_exc()}')
        except Exception as e:
            Warn(f'Prediction Error {e} @ {ix}::{traceback.format_exc()}')
            _to.touch()

        try:
            write_json(json.loads(pred), _to)
        except Exception as e:
            _to.touch()
            Warn(f'Json Dump {e} @ {ix}::{traceback.format_exc()}')

        if ix >= n:
            Info(f'Breaking Early as {n=} is set')
            break