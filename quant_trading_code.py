#%%
import os
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

# Set factors index to datetime
df_factors.index = pd.to_datetime(df_factors.index.astype(str), format='%Y%m%d')

# Returns and Live DataFrames have no headers, so we will use the stock names from df_names to set the column headers.
# We will convert the Dates DataFrame into datetime format and use these for the index of the Returns and Live DataFrames.

# Convert dates in df_dates from YYYYMMDD format to datetime.
df_dates.iloc[:, 0] = pd.to_datetime(df_dates.iloc[:, 0].astype(str), format='%Y%m%d')

# Set index of returns and live DataFrames to the dates
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

#%%
df_returns = df_returns.fillna(0)

#%%

# %%
