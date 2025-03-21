#%%
import os
import pandas as pd
import yfinance as yf

#%%

# Set directory and file paths
base_dir = os.path.dirname(__file__)
dates_path   = os.path.join(base_dir, 'datasets', 'US_Dates.xlsx')
names_path   = os.path.join(base_dir, 'datasets', 'US_Names.xlsx')

# Load  data
df_dates   = pd.read_excel(dates_path,header=None)
df_names   = pd.read_excel(names_path)

# Extract start and end dates
start_date = pd.to_datetime(df_dates.iloc[0, 0].astype(str), format='%Y%m%d')
end_date = pd.to_datetime(df_dates.iloc[-1, 0].astype(str), format='%Y%m%d')

#%%

import requests

def get_ticker_yahoo(company_name):
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {'q': company_name, 'quotesCount': 1, 'newsCount': 0}
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"Other error occurred: {err}")
        return None

    try:
        data = response.json()
    except ValueError:
        print("Failed to parse JSON. Response text:")
        print(response.text)
        return None

    quotes = data.get("quotes", [])
    if quotes:
        return quotes[0].get("symbol")
    return None


#%%
companies = list(df_names.columns)
for company in companies:
    ticker = get_ticker_yahoo(company)
    print(f"{company}: {ticker}")

#%%

# Download historical price data from Yahoo Finance using yfinance.
if ticker:
    price_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

    # Use the adjusted close prices (if available) to compute returns.
    adj_close = price_data['Adj Close']

    # Calculate percentage returns from adjusted closing prices.
    computed_returns = adj_close.pct_change()

    print("Price data downloaded for tickers:", ticker)
else:
    print("No tickers found. Check your market selection or ticker sources.")

#%%
computed_returns.head()
# %%
