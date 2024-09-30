from torch_snippets import *

dataset_info = "/home/paperspace/LLaMA-Factory/data/dataset_info.json"
js = read_json(dataset_info)
js["cord"] = {
    "file_name": "/home/paperspace/Data/cord/data.json",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}
write_json(js, dataset_info)
