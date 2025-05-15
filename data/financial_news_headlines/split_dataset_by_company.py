import json

start_date = '2025-01-01'
end_date = '2025-01-31'
data_file_path = f'data/financial_news_headlines/origin_data_{start_date}_{end_date}.json'

with open(data_file_path, 'r') as f:
    data = json.load(f)

news_id = 0
company_id = 0

store_data = []
total_news_number = 0

for company_id, company_symbol_dict in enumerate(data):
    if company_id >= 100:
        break
    store_data.append(company_symbol_dict)

    company_id += 1
    print("******")
    company_symbol = company_symbol_dict["company_symbol"]
    company_name = company_symbol_dict["company_name"]
    print(f"Company id: {company_id}, Company: {company_symbol} ({company_name})")
    print(f"News number: {len(company_symbol_dict['financial_news'])}")
    print("******")
    total_news_number += len(company_symbol_dict['financial_news'])

print(f"Tottal news number: {total_news_number}")

split_file_path = f'data/financial_news_headlines/origin_data_{start_date}_{end_date}_Top100.json'
with open(split_file_path, 'w') as f:
    json.dump(store_data, f)
