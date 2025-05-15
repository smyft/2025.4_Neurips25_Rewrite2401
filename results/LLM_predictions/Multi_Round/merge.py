import json
from datetime import datetime

current_time = "2025-04-21_18-44-41"

start_idx = 1
end_idx = 12

merge_data = []

for idx in range(start_idx, end_idx + 1):
    with open(f"./results/LLM_predictions/Multi_Round/{current_time}_Part{idx}.json", 'r') as f:
        data = json.load(f)
        merge_data.extend(data)

with open(f"./results/LLM_predictions/Multi_Round/{current_time}.json", 'w') as f:
    json.dump(merge_data, f)
