import os
import json
import numpy as np

def summarize_statistics_numpy(data):

    try:
        np_array = np.array(data, dtype=float) # Try converting to float array
    except ValueError:
        return "Input list must contain only numbers."

    n = len(np_array)

    minimum = np.min(np_array)
    maximum = np.max(np_array)
    mean = np.mean(np_array)
    median = np.median(np_array)

    # Standard deviation: ddof=1 calculates sample standard deviation (divides by n-1)
    # Note: np.std(..., ddof=1) will return NaN for a single element, which is
    # arguably more correct than 0 for sample SD. Let's handle it explicitly.
    if n >= 2:
        st_dev = np.std(np_array, ddof=1)
    else:
        st_dev = np.nan # NumPy's convention for sample SD of < 2 points

    # Percentiles
    # np.percentile uses linear interpolation by default, which is common.
    # You could specify 'nearest', 'lower', 'higher', 'midpoint' if needed.
    if n >= 1: # Percentiles are defined for N>=1
      percentile_25 = np.percentile(np_array, 25)
      percentile_75 = np.percentile(np_array, 75)
    else:
        percentile_25 = np.nan
        percentile_75 = np.nan


    summary = {
        "mean": mean,
        "sd": st_dev,
        "min": minimum,
        "P25": percentile_25,
        "median": median,
        "P75": percentile_75,
        "max": maximum
    }

    # Convert numpy values to standard Python types for the dictionary
    # (Optional, but can be cleaner)
    summary = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in summary.items()}

    return summary

data_file_path = f"./data/financial_news_headlines/origin_data_2025-01-01_2025-01-31_AAPL.json"

title_length = []
content_length = []
sentiment = []

with open(data_file_path, 'r') as f:
    data = json.load(f)

for company_symbol_dict in data:
    company_symbol = company_symbol_dict["company_symbol"]
    company_name = company_symbol_dict["company_name"]
    for news_dict in company_symbol_dict["financial_news"]:
        title = news_dict["title"]
        content = news_dict["content"]

        pos = news_dict["sentiment"]["pos"]
        neu = news_dict["sentiment"]["neu"]
        neg = news_dict["sentiment"]["neg"]

        title_length.append(len(title))
        content_length.append(len(content))
        sentiment.append(pos + neu / 2)

print("------")
summary = summarize_statistics_numpy(title_length)
print(summary)
print("------")

print("------")
summary = summarize_statistics_numpy(content_length)
print(summary)
print("------")

print("------")
summary = summarize_statistics_numpy(sentiment)
print(summary)
print("------")
