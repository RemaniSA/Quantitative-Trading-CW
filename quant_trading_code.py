#%%
import os
import matplotlib.pyplot as plt
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
# Q1: Data Preprocessing

# Set factors df index to datetime
df_factors.index = pd.to_datetime(df_factors.index.astype(str), format='%Y%m%d')

# Drop empty columns from the factors df
df_factors=df_factors.drop(columns=['Unnamed: 5', 'Unnamed: 6'])

# Returns and Live df have no headers, so we will use the stock names from df_names to set column headers
# We will convert Dates df into datetime format and use these for the index of Returns and Live df

# Convert dates in df_dates from YYYYMMDD format to datetime.
df_dates.iloc[:, 0] = pd.to_datetime(df_dates.iloc[:, 0].astype(str), format='%Y%m%d')

# Set index of returns and live df to dates
df_returns.index = df_dates.iloc[:, 0]
df_live.index = df_dates.iloc[:, 0]

# Use column headers from df_names as stock names
stock_names = df_names.columns.tolist()
df_returns.columns = stock_names
df_live.columns = stock_names

# Set index as date
df_returns.index.name = 'Date'
df_live.index.name = 'Date'

# Identify dead stocks: those with a live indicator sum of 0 across the sample period
dead_stocks = df_live.columns[df_live.sum() == 0].tolist()
print("Dead stocks:", dead_stocks)
print(f'Number of dead stocks: {len(dead_stocks)}')

# Drop Dead Stocks
df_returns=df_returns.drop(columns=dead_stocks)
df_live=df_live.drop(columns=dead_stocks)

# Print df shapes
print("Returns DataFrame shape:", df_returns.shape)
print("Returns DataFrame shape:", df_returns.shape)
print("Live DataFrame shape:", df_live.shape)
print("Dates DataFrame shape:", df_dates.shape)
print("Names DataFrame shape:", df_names.shape)
print("Factors DataFrame shape:", df_factors.shape)

#%%
# Q2: Compute Standard Momentum as Sum of Weekly Returns 
# based on Jegadeesh and Titman (1993):
# Momentum_Return(i) = Sum of weekly returns from t–48 to t–5
# This is a 44-week sum (48 - 4 = 44)

# Shift returns by 4 periods
shifted_returns = df_returns.shift(4)

# Define a function to compute sum only if no NaN values in the window
def safe_sum(x):
    # x is a numpy array of returns in the current rolling window
    if np.isnan(x).any():
        return np.nan
    else:
        return np.sum(x)

# Use rolling window of 44 weeks and apply the safe_sum function
# Setting min_periods=44 ensures we only compute sum when full window is available
df_momentum = shifted_returns.rolling(window=44, min_periods=44).apply(safe_sum, raw=True)

# Display first few rows of computed momentum factor
print("Standard Momentum Factor (sum of returns) - head:")
print(df_momentum.head())

# Save momentum factor to a CSV file
df_momentum.to_csv('datasets/US_Momentum.csv')

# %%
# Q3: Fama-MacBeth Regression
#
# In this cell, we perform weekly Fama-MacBeth regressions where we regress one-week ahead stock returns
# on one-week lagged momentum exposures (the factor from Q2). For each week, we only include stocks that:
#   1. Are "live" (i.e., have a live indicator of 1).
#   2. Have valid (non-missing) momentum and return data.
#
# We set a minimum number of stocks (min_obs) to ensure that the cross-sectional regression is based on a 
# sufficiently large sample; having too few stocks can lead to unstable estimates 
# driven by noise or outliers and unreliable t-statistics due to reduced degrees of freedom.
#
# The weekly regression coefficients (gamma) are collected, and a t-statistic is computed over these gamma values 
# to assess the overall significance of the momentum factor.


def famaMacBeth(factor, returns, live, min_obs=5):
    """
    Performs weekly Fama-MacBeth regressions.
    
    Parameters:
        factor (pd.DataFrame): DataFrame of factor exposures (e.g., momentum) 
                               with Date as index and stocks as columns.
        returns (pd.DataFrame): DataFrame of returns with Date as index and stocks as columns.
        live (pd.DataFrame): DataFrame of live indicators (1 = live, 0 = dead) 
                             with Date as index and stocks as columns.
        min_obs (int): Minimum number of stocks required to run regression for a period.
        
    Returns:
        df_gamma (pd.DataFrame): DataFrame with gamma coefficients (factor returns) indexed by date.
        tstat (float): t-statistic computed over the gamma series.
    """
    gamma_list = []
    
    # Loop over each date in the factor DataFrame
    for t in factor.index:
        # Find position of current date in returns index
        pos = returns.index.get_loc(t)
        # Ensure there's a next period (t+1) available for the dependent variable
        if pos + 1 >= len(returns.index):
            continue
        t_next = returns.index[pos + 1]
        
        # Independent variable: factor exposure at t
        x = factor.loc[t]
        # Dependent variable: one-week ahead returns at t+1
        y = returns.loc[t_next]
        # Live indicator for filtering
        live_t = live.loc[t]
        
        # Only include stocks that are live and have non-missing x and y
        valid = (live_t == 1) & x.notna() & y.notna()
        if valid.sum() < min_obs:
            continue
        
        # Subset x and y to valid stocks.
        x_valid = x[valid]
        y_valid = y[valid]
        
        # Add column of ones for intercept
        X_design = np.vstack([np.ones(len(x_valid)), x_valid]).T
        
        # Run OLS regression: y_valid = a + gamma * x_valid
        coefs, _, _, _ = np.linalg.lstsq(X_design, y_valid, rcond=None)
        gamma = coefs[1]  # Extract gamma (slope coefficient)
        
        gamma_list.append((t, gamma))
    
    # Convert results to a DataFrame
    df_gamma = pd.DataFrame(gamma_list, columns=['Date', 'Gamma']).set_index('Date')
    
    # Compute the t-statistic for the gamma series
    T = len(df_gamma)
    if T > 0:
        gamma_vals = df_gamma['Gamma'].values
        tstat = np.nanmean(gamma_vals) / (np.nanstd(gamma_vals) / np.sqrt(T))
    else:
        tstat = np.nan
        
    return df_gamma, tstat

# Run Fama-MacBeth regression using the momentum factor and stock returns
df_gamma, tstat = famaMacBeth(df_momentum, df_returns, df_live)
print("Fama–MacBeth Regression - Weekly Momentum Factor Coefficients (Gamma):")
print(df_gamma.head())
print("\nT-Statistic:", tstat)

# Save gamma coefficients to a CSV file
df_gamma.to_csv('datasets/US_FMB_Gamma.csv')

# Plot the factor returns
plt.figure(figsize=(10,6))
plt.plot(df_gamma)
plt.title('Factor Returns')

# %%
# Grid Search for Minimum Observation Threshold (min_obs)
#
# In this section, we search for the lowest threshold of valid observations 
# for which the Fama–MacBeth regression produces a statistically significant t-statistic (|t-stat| >= 1.96).
#
# First, we calculate the number of valid observations per week. 
# Here, a "valid observation" is defined as a stock that is "live" (live indicator equals 1) 
# and has a non-missing momentum factor value.
#
# We then review the median and key quantiles of these counts to determine a reasonable grid range 
# for the minimum number of observations required in our regressions.


valid_counts = []

for t in df_momentum.index:
    valid = (df_live.loc[t] == 1) & (df_momentum.loc[t].notna())
    valid_counts.append(valid.sum())

valid_counts_series = pd.Series(valid_counts, index=df_momentum.index)

# Print summary statistics for valid observation counts
print("Summary of Valid Observation Counts per Week:")
print(valid_counts_series.describe())
print("\nMedian valid observations:", valid_counts_series.median())
print("\nQuantiles (25%, 50%, 75%):")
print(valid_counts_series.quantile([0.25, 0.5, 0.75]))

grid_min_obs = np.arange(800, 2000, 100)
first_significant = None

for min_obs in grid_min_obs:
    df_gamma_temp, tstat_temp = famaMacBeth(df_momentum, df_returns, df_live, min_obs=min_obs)
    print(f"Minimum obs = {min_obs}: t-statistic = {tstat_temp:.4f}")
    if np.abs(tstat_temp) >= 1.96:
        first_significant = min_obs
        print(f"First significant t-stat found with min_obs = {min_obs}, t-statistic = {tstat_temp:.4f}")
        break

if first_significant is None:
    print("No significant t-statistic found within the grid of minimum observations.")