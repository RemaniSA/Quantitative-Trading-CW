#%%
import os
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#%%
# Data Loading

# Set directory and file paths
base_dir = os.path.dirname(__file__)
returns_path = os.path.join(base_dir, 'datasets', 'US_Returns.csv')
live_path  = os.path.join(base_dir, 'datasets', 'US_live.csv')
dates_path = os.path.join(base_dir, 'datasets', 'US_Dates.xlsx')
names_path = os.path.join(base_dir, 'datasets', 'US_Names.xlsx')
factors_path = os.path.join(base_dir, 'datasets', 'FamaFrench.csv')

# Load  data
df_returns = pd.read_csv(returns_path,header=None)
df_live = pd.read_csv(live_path,header=None)
df_dates = pd.read_excel(dates_path,header=None)
df_names = pd.read_excel(names_path)
df_factors=pd.read_csv(factors_path,index_col='Unnamed: 0')

#%% 
# Data Preprocess

# Set factors df index to datetime
df_factors.index = pd.to_datetime(df_factors.index.astype(str), format='%Y%m%d')

# Drop empty columns from the factors df
df_factors=df_factors.drop(columns=['Unnamed: 5', 'Unnamed: 6'])

# Returns and Live df have no headers, so we will use the stock names from df_names to set the column headers.
# We will convert the Dates df into datetime format and use these for the index of the Returns and Live DataFrames.

# Convert dates in df_dates from YYYYMMDD format to datetime.
df_dates.iloc[:, 0] = pd.to_datetime(df_dates.iloc[:, 0].astype(str), format='%Y%m%d')

# Set index of returns and live df to the dates
df_returns.index = df_dates.iloc[:, 0]
df_live.index = df_dates.iloc[:, 0]

# Use the column headers from df_names as the stock names
stock_names = df_names.columns.tolist()
df_returns.columns = stock_names
df_live.columns = stock_names

# Set index as date
df_returns.index.name = 'Date'
df_live.index.name = 'Date'

# Identify dead stocks: those with a live indicator sum of 0 across the sample period.
dead_stocks = df_live.columns[df_live.sum() == 0].tolist()
print("Dead stocks:", dead_stocks)
print(f'Number of dead stocks: {len(dead_stocks)}')

# Drop Dead Stocks
df_returns=df_returns.drop(columns=dead_stocks)
df_live=df_live.drop(columns=dead_stocks)

# Print DataFrame shapes for verification
print("Returns DataFrame shape:", df_returns.shape)
print("Returns DataFrame shape:", df_returns.shape)
print("Live DataFrame shape:", df_live.shape)
print("Dates DataFrame shape:", df_dates.shape)
print("Names DataFrame shape:", df_names.shape)
print("Factors DataFrame shape:", df_factors.shape)

#%%
# Q2: Compute Standard Momentum as a Sum of Weekly Returns 
# based on Jegadeesh and Titman (1993):
# Momentum_Return(i) = Sum of weekly returns from t–48 to t–5.
# (This gives you a 44-week sum when shifting by 4 weeks, since 48 - 4 = 44.)

# Shift returns by 4 periods to exclude the most recent 4 weeks
shifted_returns = df_returns.shift(4)

# Define a function to compute sum only if there are no NaN values in the window
def safe_sum(x):
    # x is a numpy array of returns in the current rolling window
    if np.isnan(x).any():
        return np.nan
    else:
        return np.sum(x)

# Use a rolling window of 44 weeks (i.e. from t-48 to t-5) and apply the safe_sum function.
# Setting min_periods=44 ensures we only compute the sum when the full window is available.
df_momentum = shifted_returns.rolling(window=44, min_periods=44).apply(safe_sum, raw=True)

# Display the first few rows of the computed momentum factor
print("Standard Momentum Factor (sum of returns) - head:")
print(df_momentum.head())

# %%
df_momentum.to_csv('datasets/US_Momentum.csv')

# %%
