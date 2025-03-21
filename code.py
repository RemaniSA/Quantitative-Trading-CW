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
live_path    = os.path.join(base_dir, 'datasets', 'US_live.csv')
dates_path   = os.path.join(base_dir, 'datasets', 'US_Dates.xlsx')
names_path   = os.path.join(base_dir, 'datasets', 'US_Names.xlsx')

# Load  data
df_returns = pd.read_csv(returns_path,header=None)
df_live    = pd.read_csv(live_path,header=None)
df_dates   = pd.read_excel(dates_path,header=None)
df_names   = pd.read_excel(names_path)

#%% 
# Data Preprocess

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
print("Live DataFrame shape:", df_live.shape)
print("Dates DataFrame shape:", df_dates.shape)
print("Names DataFrame shape:", df_names.shape)

#%%
# Identify missing returns for live stocks: where df_live==1 and df_returns is NaN.
missing_live_returns = (df_live == 1) & (df_returns.isna())


#%%
# If there are any such issues, print a warning and the problematic values.
if missing_live_returns.any().any():
    print("Warning: Missing returns for live stocks detected!")
    print(df_returns[missing_live_returns])

    # Stack the DataFrame to get a Series with a MultiIndex (Date, Stock)
    missing_series = missing_live_returns.stack()

    # Filter to only the True values
    missing_series = missing_series[missing_series]

    # Group by stock (the second level of the MultiIndex) and collect the dates into a list
    missing_by_stock = missing_series.groupby(level=1).apply(lambda x: x.index.get_level_values(0).tolist())

    print(missing_by_stock)

# # Fill all remaining missing values in df_returns with 0 (assuming no return means 0)
# df_returns = df_returns.fillna(0)


# %%
