Supplementary code for the blog - https://nanonets.com/blog/vision-language-model-vlm-for-data-extraction

## Usage

Create a new conda environment (Code was written on python3.12)
```bash
pip install -r requirements.txt
pip install -e .
```

```python
from vlm.available_models import available_models

available_models()
# Qwen2, GPT4oMini, Claude_35, MiniCPM, ...
```

```python
from vlm.available_models import VLMs

print(VLMs) # --> Dictionary of VLM classes
# you can pick any one of them and use it like so
qwen = VLMs['Qwen2']()
image = '/path/to/image.jpg'
prompt = 'what is in the picture'
prediction = qwen(image, prompt)
```

Predictions are cached internally in `db/predictions.db` and all the VLMs will avoid redundant API/Model calls on same inputs

```python
with duckdb.connect('db/predictions.db') as con:
    q = f"SELECT * FROM Predictions"
    df = con.execute(q).fetchdf()
print(df.columns)
# Index(['inputs_hash', 'prompt', 'kwargs', 'vlm_name', 'dataset_name',
#        'dataset_row_index', 'prediction_value', 'prediction_duration',
#        'error_string'],
#       dtype='object')
```
