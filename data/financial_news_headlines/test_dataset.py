import json

start_date = '2025-01-01'
end_date = '2025-01-31'
data_file_path = f'data/financial_news_headlines/origin_data_{start_date}_{end_date}_Top100.json'

with open(data_file_path, 'r') as f:
    data = json.load(f)

news_id = 0
company_id = 0
for company_symbol_dict in data:
    company_symbol = company_symbol_dict["company_symbol"]
    company_name = company_symbol_dict["company_name"]
    for news_dict in company_symbol_dict["financial_news"]:
        headline = news_dict["title"]
        content = news_dict["content"]
        news_id += 1

        assert isinstance(content, str)

        print("******")
        print(f"News id: {news_id}")
        print(f"Haedline: {headline}")
        print("******")
    
    company_id += 1
    print("******")
    print(f"Company id: {company_id}, Company: {company_symbol} ({company_name})")
    print(f"News number: {len(company_symbol_dict['financial_news'])}")
    print("******")

    # break

print(f"Tottal news number: {news_id}")
