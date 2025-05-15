import json

import pandas as pd
import requests
from requests.exceptions import RequestException

import time

start_date = '2025-01-01'
end_date = '2025-01-31'

api_token = " 65b220b26ae3b4.21151736"

company_name_file_path = './data/financial_news_headlines/S&P_500_info.csv'
df = pd.read_csv(company_name_file_path, na_filter=False)
symbol_list = df['Symbol']
# symbol_list = ["AAPL"]
name_list = [df.loc[df['Symbol']==symbol, 'Company'].values[0] for symbol in symbol_list]

file_part_idx = 4
cur_idx = 93

while cur_idx < len(symbol_list):
    with open(f'data/financial_news_headlines/origin_data_{start_date}_{end_date}_part{file_part_idx}.json', 'w') as f:
        news_headlines_json_format = []
        entries_count = 0

        while cur_idx < len(symbol_list):
            symbol = symbol_list[cur_idx]
            name = name_list[cur_idx]

            print(f"\nFetching data for {symbol} ({name})")

            try:
                url = f'https://eodhd.com/api/news?s={symbol}&from={start_date}&to={end_date} \
                    &offset=0&api_token={api_token}&limit={1000}&fmt=json'
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
            except (RequestException, json.JSONDecodeError) as e:
                print(f"API request {cur_idx} failed. Company symbol: {symbol}, company name: {name}")
                print(f"Error message: {str(e)}")
                break
        
            origin_data = {"company_symbol": symbol, "company_name":name, "financial_news": data}
            news_headlines_json_format.append(origin_data)
            entries_count += len(data)

            cur_idx += 1
    
        print(f"\nTotal entries collected: {entries_count}")
        json.dump(news_headlines_json_format, f)
        file_part_idx += 1

        time.sleep(30)


'''
url = f'https://eodhd.com/api/sentiments?s=aapl.us&from={start_date}&to={end_date}&api_token=demo&fmt=json'
data = requests.get(url).json()
print(data)
'''
