import json
from datetime import datetime

current_time = "2025-05-05_17-14-05"

start_idx = 1
end_idx = 8

merge_data = []

for idx in range(start_idx, end_idx + 1):
    with open(f"./results/LLM_predictions/Multi_Round_Sampling/{current_time}_Part{idx}.json", 'r') as f:
        data = json.load(f)

        company_id = data[0]["company_id"]
        company_symbol = data[0]["company_symbol"]
        company_name = data[0]["company_name"]
        company_predictions = data[0]["predictions"]
        
        merge_data.extend(company_predictions)

with open(f"./results/LLM_predictions/Multi_Round_Sampling/{current_time}.json", 'w') as f:
    result = [{
        "company_id": company_id,
        "company_symbol": company_symbol,
        "company_name": company_name,
        "predictions": merge_data
    }]
    
    json.dump(result, f)
