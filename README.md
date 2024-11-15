Supplementary code for the blogs -   
https://nanonets.com/blog/vision-language-model-vlm-for-data-extraction  
and  
https://nanonets.com/blog/fine-tuning-vision-language-models-vlms-for-data-extraction

## Inference

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
from vlm.available_vlms import VLMs

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
#        'item_name', 'prediction_value', 'prediction_duration',
#        'error_string'],
#       dtype='object')
```


## Fine Tuning
#### Step 1: Clone the training engine
```bash
git clone https://github.com/hiyouga/LLaMA-Factory/ /home/paperspace/LLaMA-Factory/
```

#### Step 2: Create cord in sharegpt format
```bash
vlm save-cord-dataset-in-sharegpt-format /home/paperspace/Data/cord/data.json
```

#### Step 3: Register the created dataset in LLaMA-Factory
```bash
./scripts/register-cord-dataset.py
```

#### Step 4: Create a configuration yaml for training CORD with QWEN2
```bash
cd /home/paperspace/LLaMA-Factory/
```

Create a new file in examples/train_lora/cord.yaml in LLama-Factory folder as given below

```yaml
### model
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: cord
template: qwen2_vl
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/cord-4/qwen2_vl-2b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 8
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 10.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

#### Step 5: Train the model using llamafactory-cli with the YAML file created above

```bash
llamafactory-cli train examples/train_lora/cord.yaml
```

#### Step 6: Predict with the trained model
```bash
vlm predict-cord Qwen2_Custom --vlm-kwargs='{"model": "Qwen/Qwen2-VL-2B-Instruct", "name": "Qwen-2B-Cord-v4-Finetuned", "adapter": "/home/paperspace/LLaMA-Factory/saves/cord-4/qwen2_vl-2b/lora/sft"}'
```

#### Step 7: Evaluate the accuracies of the trained model on CORD dataset
```bash
vlm evaluate-cord-cli Qwen-2B-Cord-v3-Finetuned
```

#### Step 8: Compare results of above model with old model
```bash
vlm evaluate-cord-cli Qwen2
```


