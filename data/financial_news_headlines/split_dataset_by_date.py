import json
from datetime import datetime

start_date = '2025-01-01'
end_date = '2025-01-31'
data_file_path = f'data/financial_news_headlines/origin_data_{start_date}_{end_date}_AAPL.json'
split_data_file_path = f'data/financial_news_headlines/origin_data_{start_date}_{end_date}_AAPL_split.json'

# data_file_path = f'data/financial_news_headlines/origin_data_{start_date}_{end_date}_Top100.json'
# split_data_file_path = f'data/financial_news_headlines/origin_data_{start_date}_{end_date}_Top100_split.json'

with open(data_file_path, 'r') as f:
    data = json.load(f)

news_id = 0
company_id = 0
store_data = []
for company_id, company_symbol_dict in enumerate(data):
    company_symbol = company_symbol_dict["company_symbol"]
    company_name = company_symbol_dict["company_name"]

    company_data = []
    current_date = None
    current_date_data = []

    for news_dict in company_symbol_dict["financial_news"]:
        dt = datetime.fromisoformat(news_dict["date"])
        dt = dt.date().isoformat()
        if dt != current_date:
            if current_date is not None:
                company_data.append({"date": current_date, "news": current_date_data})
            current_date = dt
            current_date_data = [news_dict]
        else:
            current_date_data.append(news_dict)

        news_id += 1
    
    store_data.append({
        "company_symbol": company_symbol,
        "company_name": company_name,
        "financial_news": company_data
    })

    print("******")
    print(f"Company id: {company_id}, Company: {company_symbol} ({company_name})")
    print(f"News number: {len(company_symbol_dict['financial_news'])}")
    print("******")

    # break

print(f"Tottal news number: {news_id}")

with open(split_data_file_path, 'w') as f:
    json.dump(store_data, f)
