import duckdb, json
from torch_snippets import *
from datasets import load_dataset

from vlm.cli import cli
from vlm.available_vlms import VLMs
from vlm.data.base import main, process_raw
from vlm.evaluation.exact_match import ExactMatch


def load_sroie(split="test"):
    ds = load_dataset("sizhkhy/SROIE", split=split)
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
def evaluate_sroie(vlm: str, db: P = None):
    db = ifnone(db, os.environ["DUCKDB"])
    with duckdb.connect(db) as con:
        q = f"SELECT dataset_row_index, prediction_value, error_string, vlm_name FROM predictions where dataset_name = 'SROIE' and vlm_name = '{vlm}'"
        df = con.execute(q).fetchdf()
        df = df[~df["prediction_value"].isna()]
        df = df[df["prediction_value"] != None]

    metrics = AD()
    ds, fields = load_sroie()

    import evaluate
    from evaluate.module import Metric

    for f in fields:
        metrics[f] = ExactMatch()

    for _, (_, row) in E(track2(df.iterrows(), total=len(df))):
        row = row.squeeze()
        pred = AD(process_raw(VLMs[row.vlm_name].get_raw_output(row.prediction_value)))
        pred = AD({k.upper(): v for k, v in pred.items()})
        err = row.error_string
        ix = row.dataset_row_index
        truth = ds[ix]["fields"]
        for f in fields:
            m: Metric = metrics[f]
            _pred = str(pred.get(f, "")).replace("\n", " ")
            _truth = str(truth[f])
            m.add_batch(predictions=[_pred], references=[_truth])
    aggregate = AD()
    cache = AD()
    for _m in metrics:
        m: Metric = metrics[_m]
        m._finalize()
        _cache = m.data
        _aggregate = _agg = m.compute(ignore_case=True, ignore_punctuation=True)
        aggregate[_m] = float(_agg["exact_match"])
        cache[_m] = _cache.add_column("scores", _agg["score_list"]).to_pandas()

    return AD(aggregate=aggregate, cache=cache)


if __name__ == "__main__":
    cli()
