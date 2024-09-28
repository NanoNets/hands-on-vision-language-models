import json, duckdb
from torch_snippets import *
from datasets import load_dataset

from vlm.cli import cli
from vlm.available_vlms import VLMs
from vlm.data.base import main, FieldMatcher, process_raw
from vlm.evaluation.grits import grits_con


def load_cord(split='test'):
    ds = load_dataset("naver-clova-ix/cord-v2", split=split)
    return ds


def load_gt(x):
    x = AD(x)
    gt = AD(json.loads(x.ground_truth)).gt_parse
    return gt


def get_table_total_subtotal_fields():
    TABLE_FIELDS, TOTAL_FIELDS, SUBTOTAL_FIELDS = [], [], []
    ds = load_cord()

    for x in track2(ds):
        gt = load_gt(x)
        menu = gt.menu
        sub_total, total = gt.get("sub_total", {}), gt.get("total", {})

        if isinstance(menu, AD):
            TABLE_FIELDS.extend(list(menu.keys()))
        else:
            table_fields = flatten([l.keys() for l in menu])
            TABLE_FIELDS.extend(table_fields)

        SUBTOTAL_FIELDS.extend(list(sub_total.keys()))
        TOTAL_FIELDS.extend(list(total.keys()))

    TABLE_FIELDS, TOTAL_FIELDS, SUBTOTAL_FIELDS = [
        Counter(d) for d in [TABLE_FIELDS, TOTAL_FIELDS, SUBTOTAL_FIELDS]
    ]
    return TABLE_FIELDS, TOTAL_FIELDS, SUBTOTAL_FIELDS

ALL_FIELDS = AD(
    TABLE_FIELDS = 'nm,price,cnt,unitprice'.split(','),
    SUBTOTAL_FIELDS = 'subtotal_price,tax_price'.split(','),
    TOTAL_FIELDS = 'total_price,cashprice,changeprice'.split(','),
)

prompt = """Extract the following data from given image - 

For tables I need a json of list of 
dictionaries of following keys per dict (one dict per line)
'nm', # name of the item
'price', # total price of all the items combined
'cnt', # quantity of the item
'unitprice' # price of a single igem

For sub-total I need a single json of
{'subtotal_price', 'tax_price'}

For total I need a single json of
{'total_price', 'cashprice', 'changeprice'}

the final output should look like and must be JSON parsable
{
    "menu": [
        {"nm": ..., "price": ..., "cnt": ..., "unitprice": ...}
        ...
    ],
    "subtotal": {"subtotal_price": ..., "tax_price": ...},
    "total": {"total_price": ..., "cashprice": ..., "changeprice": ...}
}
If a field is missing,
simply omit the key from the dictionary. Do not infer.
Return only those values that are present in the image.
this applies to highlevel keys as well, i.e., menu, subtotal and total
"""


@cli.command()
def predict_cord(vlm: str, n: int = None):
    """VLM should one of the available VLMs"""
    ds = load_cord()
    main(ds, vlm, prompt, n, image_key="image", dataset_name="CORD")


@cli.command()
def evaluate_cord(vlm: str, db: P=None):
    """db is the folder where the predictions were saved"""
    db = ifnone(db, os.environ['DUCKDB'])
    with duckdb.connect(db) as con:
        q = f"SELECT dataset_row_index, prediction_value, error_string, vlm_name FROM predictions where dataset_name = 'CORD' and vlm_name = '{vlm}'"
        df = con.execute(q).fetchdf()
        df = df[~df['prediction_value'].isna()]
        df = df[df['prediction_value'] != None]

    ds = load_cord()

    field_metrics = FieldMatcher([*ALL_FIELDS.SUBTOTAL_FIELDS, *ALL_FIELDS.TOTAL_FIELDS])
    _results, table_metrics = AD(), []
    for _, row in track2(df.iterrows(), total=len(df)):
        pred = AD(process_raw(VLMs[row.vlm_name].get_raw_output(row.prediction_value)))
        ix = int(row.dataset_row_index)
        truth = load_gt(ds[ix])
        _truth_fields = AD({**truth.get('total', {}), **truth.get('subtotal', {})})
        _pred_fields = AD({**pred.get('total', {}), **pred.get('subtotal', {})})
        field_metrics.update(truth=_truth_fields, pred=_pred_fields)
        row = {**row, **evaluate_table(truth, pred)}
        table_metrics.append(row)

    _results['fields'] = field_metrics.compute()
    table_metrics = pd.DataFrame(table_metrics)
    aggregate = table_metrics[['precision','recall','fscore']].mean().to_dict()
    _results['table'] = AD(cache=table_metrics, aggregate=aggregate)
    return _results

def evaluate_table(truth, pred):
    truth_menu = truth.menu
    truth_menu = [truth_menu] if not isinstance(truth_menu, (L,list)) else list(truth_menu)

    pred = pd.DataFrame([p.d for p in pred.get('menu', AD())]).map(lambda x: str(x).lower().replace(',', ''))
    truth = pd.DataFrame([t.d for t in truth_menu]).map(lambda x: str(x).lower().replace(',', ''))

    columns = sorted([c for c in pred.columns if not (pred[c].nunique() == 1 and pred[c].unique()[0] == 'None')])
    pred = pred[[c for c in columns if c in ALL_FIELDS.TABLE_FIELDS]]

    columns = sorted([c for c in truth.columns if not (truth[c].nunique() == 1 and truth[c].unique()[0] == 'None')])
    truth = truth[[c for c in columns if c in ALL_FIELDS.TABLE_FIELDS]]

    if in_debug_mode():
        show(pred)
        show(truth)

    fscore, precision, recall, _ = grits_con(pred.values, truth.values)
    return AD(fscore=fscore, precision=precision, recall=recall)


    


if __name__ == "__main__":
    cli()
