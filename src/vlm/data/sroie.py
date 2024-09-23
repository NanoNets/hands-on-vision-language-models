import json
from torch_snippets import *

from vlm.data.base import main
from vlm.cli import cli


def load_sroie():
    from datasets import load_dataset

    ds = load_dataset("sizhkhy/SROIE", split="test")
    fields = ds[0]["fields"].keys()
    return ds, fields


@cli.command()
def predict_sroie(vlm: str, n: int = None):
    """VLM should one of the available VLMs. db_folder is the folder where the predictions will be saved"""
    ds, fields = load_sroie()
    _fields = ",".join(fields)
    prompt = f"What are the {_fields} in the given image. Give me as a json"
    main(ds, vlm, prompt, n, dataset_name="SROIE")


@cli.command()
def evaluate_sroie(db: P):
    """db is the folder where the predictions were saved"""
    metrics = AD()
    ds, fields = load_sroie()

    import evaluate

    for f in fields:
        metrics[f] = evaluate.load("exact_match")

    for ix, file in E(track2(db.ls())):
        pred = read_json(file)
        pred = {k.upper(): v for k, v in pred.items()}
        truth = ds[int(file.stem)]["fields"]
        for f in fields:
            m = metrics[f]
            _pred = str(pred[f]).replace("\n", " ")
            _truth = str(truth[f])
            m.add_batch(predictions=[_pred], references=[_truth])
    aggregate = AD()
    for m in metrics:
        aggregate[m] = float(
            metrics[m].compute(ignore_case=True, ignore_punctuation=True)["exact_match"]
        )
    return aggregate


if __name__ == "__main__":
    cli()
