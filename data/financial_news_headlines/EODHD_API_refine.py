import json
import time
import pandas as pd
import requests
from requests.exceptions import RequestException

start_date = '2025-01-01'
end_date = '2025-01-31'

api_token = " 65b220b26ae3b4.21151736"

company_name_file_path = './data/financial_news_headlines/company_name_and_symbol.csv'
df = pd.read_csv(company_name_file_path, na_filter=False)
symbol_list = df['Symbol']
name_list = [df.loc[df['Symbol']==symbol, 'Company Name'].values[0] for symbol in symbol_list]

def make_api_request(url):
    retry_count = 0
    base_delay = 1  # seconds
    max_reties = 5
    
    while True:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Validate response content
            if isinstance(data, list) and len(data) > 0:
                return data
                
            print(f"Empty response, retrying...")
            
        except (RequestException, json.JSONDecodeError) as e:
            print(f"API request failed: {str(e)}")
        
        sleep_time = min(base_delay * (2 ** retry_count), base_delay * (2 ** max_reties))
        print(f"Retry {retry_count+1} in {sleep_time}s")
        time.sleep(sleep_time)
        retry_count += 1

with open(f'data/financial_news_headlines/origin_data_{start_date}_{end_date}.json', 'w') as f:
    news_headlines_json_format = []
    for symbol, name in zip(symbol_list, name_list):
        print(f"\nFetching data for {symbol} ({name})")
        url = f'https://eodhd.com/api/news?s={symbol}&from={start_date}&to={end_date}&offset=0&api_token={api_token}&limit=1&fmt=json'
        
        try:
            data = make_api_request(url)
            origin_data = {
                "company_symbol": symbol,
                "company_name": name,
                "financial_news": data
            }
            news_headlines_json_format.append(origin_data)
        except Exception as e:
            print(f"Critical error for {symbol}: {str(e)}")
            continue

    print(f"\nTotal entries collected: {len(news_headlines_json_format)}")
    json.dump(news_headlines_json_format, f)
