import json
from torch_snippets import *

from vlm.data.base import main
from vlm.cli import cli


def load_cord():
    from datasets import load_dataset

    ds = load_dataset("naver-clova-ix/cord-v2", split="test")
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
def evaluate_sroie(db: P):
    """db is the folder where the predictions were saved"""
    metrics = AD()
    ds = load_cord()
    return


if __name__ == "__main__":
    cli()
