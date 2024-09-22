from .apis import *
from .models import *

from vlm.cli import cli

VLMs = {
    "gpt4omini": GPT4oMini,
    "gemini": Gemini,
    "claude_35": Claude_35,
    "florence2": Florence2,
    "qwen2": Qwen2,
    "minicpm": MiniCPM,
}

@cli.command()
def available_vlms(): print('- ' + '\n - '.join(VLMs.keys()))