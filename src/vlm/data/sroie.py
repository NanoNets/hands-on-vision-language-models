from torch_snippets import *

from vlm.data.base import main
from vlm.cli import cli

def load_sroie():
    from datasets import load_dataset
    ds = load_dataset('sizhkhy/SROIE', split='test')
    return ds

@cli.command()
def predict_sroie(vlm:str, db_folder:P, n:int=None):
    """VLM should one of the available VLMs. db_folder is the folder where the predictions will be saved"""
    ds = load_sroie()
    fields = ','.join(ds[0]['fields'].keys())
    prompt = f'What are the {fields} in the given image. Give me as a json'
    main(ds, vlm, prompt, db_folder, n)

if __name__ == "__main__":
    cli()