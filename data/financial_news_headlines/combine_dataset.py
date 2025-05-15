import json
import pandas as pd

start_date = '2025-01-01'
end_date = '2025-01-31'
start_file_idx = 1
end_file_idx = 10

company_name_file_path = './data/financial_news_headlines/S&P_500_info.csv'
df = pd.read_csv(company_name_file_path, na_filter=False)
symbol_list = df['Symbol']
name_list = [df.loc[df['Symbol']==symbol, 'Company'].values[0] for symbol in symbol_list]

company_idx = 0
entries_count = 0

data = []
for file_idx in range(start_file_idx, end_file_idx + 1):
    file_name = f'data/financial_news_headlines/origin_data_{start_date}_{end_date}_part{file_idx}.json'
    
    with open(file_name, 'r') as f:
        data_idx = json.load(f)

        for company_dict in data_idx:
            symbol = company_dict["company_symbol"]
            company_name = company_dict["company_name"]
            if (symbol != symbol_list[company_idx]) or (company_name != name_list[company_idx]):
                print(f"\nError at idx {company_idx}")
                print(f"List: {symbol_list[company_idx]} ({name_list[company_idx]})")
                print(f"File: {symbol} ({company_name})")
            company_idx += 1

            entries_count += len(company_dict["financial_news"])
        
        data.extend(data_idx)

print(f"\nTotal entries collected: {entries_count}")

combine_file_path = f'data/financial_news_headlines/origin_data_{start_date}_{end_date}.json'
with open(combine_file_path, 'w') as f:
    json.dump(data, f)
