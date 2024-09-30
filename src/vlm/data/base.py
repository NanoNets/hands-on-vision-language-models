import json, traceback, time
from torch_snippets import *
from vlm.available_vlms import VLMs
from vlm.base import VLM
from vlm.evaluation.exact_match import ExactMatch


def main(
    ds,
    vlm: str | VLM,
    prompt,
    n: int = None,
    image_key="images",
    dataset_name=None,
    vlm_kwargs=None,
):
    if isinstance(vlm, str):
        assert vlm in VLMs, f"VLM {vlm} not found. Available VLMs: {list(VLMs.keys())}"
        vlm_kwargs = ifnone(vlm_kwargs, "{}")
        vlm_kwargs = json.loads(vlm_kwargs)
        vlm: VLM = VLMs[vlm](**vlm_kwargs)

    _vlm = vlm.name

    n = len(ds) if n is None else n
    for ix, x in E(track2(ds)):
        if _vlm == "Gemini":
            time.sleep(10)
        # Use vlm.cursor to fetch the actual predictions
        vlm(
            x[image_key],
            prompt=prompt,
            dataset_name=dataset_name,
            item_name=ix,
            overwrite_cache=False,
        )
        if ix >= n:
            Info(f"Breaking Early as {n=} is set")
            break


def process_raw(pred):
    default_pred = defaultdict(lambda: "")
    if pred is None:
        return default_pred

    pattern = r"\{[\s\S]*\}"
    match = re.search(pattern, pred)
    if match:
        pred = match.group(0)
    pred = pred.lstrip("```json").rstrip("```")

    try:
        pred = json.loads(pred)
        pred = {k: v for k, v in pred.items()}
    except:
        _sep = f"\n{'-'*25}\n"
        Warn(f"Failed to JSON parse{_sep}{type(pred)}{_sep}{pred}{_sep}")
        pred = default_pred

    return pred


class FieldMatcher:
    def __init__(self, fields):
        self.fields = fields
        self.metrics = AD()
        for f in fields:
            self.metrics[f] = ExactMatch()

    def update(self, *, truth, pred):
        for f in self.fields:
            m: Metric = self.metrics[f]
            _pred = str(pred.get(f, None)).replace("\n", " ")
            _truth = str(truth.get(f, None))
            m.add_batch(predictions=[_pred], references=[_truth])

    def compute(self):
        aggregate, cache = AD(), AD()
        for f in self.fields:
            m = self.metrics[f]
            m._finalize()
            _cache = m.data
            _agg = m.compute(ignore_case=True, ignore_punctuation=True)
            aggregate[f] = float(_agg["exact_match"])
            cache[f] = _cache.add_column("scores", _agg["score_list"]).to_pandas()
        return AD(aggregate=aggregate, cache=cache)
