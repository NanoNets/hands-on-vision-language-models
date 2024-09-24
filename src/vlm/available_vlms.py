from .apis import *
from .models import *

from vlm.cli import cli

VLMs = {c.__name__: c for c in [GPT4oMini,Gemini,Claude_35,Florence2,Qwen2,MiniCPM]}

@cli.command()
def available_vlms(): print('- ' + '\n - '.join(VLMs.keys()))