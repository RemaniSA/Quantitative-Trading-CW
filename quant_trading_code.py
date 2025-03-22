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


def famaMacBeth(factor, returns, live, min_obs=1800):
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

mean_factor_return = df_gamma.mean()
print(f'Mean Factor Return: {mean_factor_return[0]:.4f}')

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

# %%
# Q4: Compute Comomentum Measure
#
# We compute comomentum as per Lou and Polk (2021) (without industry adjustment)
# For each week t (starting from when a full 52-week window is available):
#   1. For each stock that is live at t, run a 52-week OLS regression:
#         r_i = alpha + beta1*(Mkt-RF) + beta2*(SMB) + beta3*(HML) + epsilon_i
#      and obtain the residual series for that stock.
#   2. At time t, rank stocks by momentum (from Q2).
#   3. Define "losers" decile as bottom 10% and "winners" decile as top 10%.
#   4. For each decile, compute pairwise Pearson correlation matrix of the regression residuals,
#      then average off-diagonal elements to obtain the winners and losers comomentum
#   5. Average these to get overall comomentum for week t.
#   6. Repeat for each week to generate a time series of comomentum.

# List to store comomentum values (one per week)
comomentum_list = []

# Set regression window size (52 weeks)
window_size = 52

# Loop over each week from the 52nd observation onward
for pos in range(window_size - 1, len(df_returns.index)):
    current_date = df_returns.index[pos]
    # Define the rolling window (past 52 weeks, including current_date)
    window_dates = df_returns.index[pos - window_size + 1 : pos + 1]
    
    # Dictionary to store each stock's residual series over the window
    residuals_dict = {}
    
    # Loop over each stock
    for stock in df_returns.columns:
        # Only consider the stock if it is live at current_date
        if df_live.loc[current_date, stock] != 1:
            continue
        
        # Get the 52-week return series for the stock
        y = df_returns.loc[window_dates, stock].values
        if np.isnan(y).any():
            continue
        
        # Get the corresponding factor data for the same window
        try:
            X = df_factors.loc[window_dates, ['Mkt-RF', 'SMB', 'HML']].values
        except KeyError:
            continue
        if np.isnan(X).any():
            continue
        
        # Add column of ones for intercept
        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Run OLS regression: y = alpha + beta * factors
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        # Compute residuals: epsilon = y - X_design.dot(beta)
        residual_series = y - X_design.dot(beta)
        residuals_dict[stock] = residual_series
    
    # If not enough stocks for decile sorting, skip this date
    if len(residuals_dict) < 10:
        comomentum_list.append((current_date, np.nan))
        continue
    
    # Create a DataFrame for residuals (columns: stocks, index: window_dates)
    df_residuals = pd.DataFrame(residuals_dict, index=window_dates)
    
    # At the current date, get the momentum measure for valid stocks
    # (df_momentum was computed in Q2)
    current_mom = df_momentum.loc[current_date]
    # Keep only stocks for which we have residuals
    valid_stocks = [stock for stock in df_residuals.columns if stock in current_mom.index and not pd.isna(current_mom[stock])]
    if len(valid_stocks) < 10:
        comomentum_list.append((current_date, np.nan))
        continue
    # Extract momentum values for valid stocks
    mom_series = current_mom[valid_stocks]
    
    # Determine decile thresholds
    lower_thresh = np.percentile(mom_series, 10)
    upper_thresh = np.percentile(mom_series, 90)
    
    # Identify loser (bottom decile) and winner (top decile) stocks
    losers = mom_series[mom_series <= lower_thresh].index.tolist()
    winners = mom_series[mom_series >= upper_thresh].index.tolist()
    
    # Function to compute average off-diagonal correlation in a residual DataFrame
    def avg_offdiag_corr(df_group):
        n = df_group.shape[1]
        if n < 2:
            return np.nan
        corr_matrix = df_group.corr().to_numpy()
        # Sum of off-diagonals = total sum minus sum of diagonal elements (which are 1)
        sum_offdiag = corr_matrix.sum() - n
        num_offdiag = n * (n - 1)
        return sum_offdiag / num_offdiag
    
    # Compute decile-specific comomentum
    comomentum_l = avg_offdiag_corr(df_residuals[losers]) if len(losers) >= 2 else np.nan
    comomentum_w = avg_offdiag_corr(df_residuals[winners]) if len(winners) >= 2 else np.nan
    
    # Compute overall comomentum as the average of the loser and winner measures
    if np.isnan(comomentum_l) or np.isnan(comomentum_w):
        overall_comomentum = np.nan
    else:
        overall_comomentum = 0.5 * (comomentum_l + comomentum_w)
    
    # Append the result for the current date
    comomentum_list.append((current_date, overall_comomentum))

# Convert results to a DataFrame
df_comomentum = pd.DataFrame(comomentum_list, columns=['Date', 'Comomentum']).set_index('Date')

print("Comomentum Measure (first few rows):")
print(df_comomentum.head())

# Optionally, save the comomentum measure to a CSV file
df_comomentum.to_csv('datasets/US_Comomentum.csv')

# Plot the comomentum measure over time
plt.figure(figsize=(10,6))
plt.plot(df_comomentum.index, df_comomentum['Comomentum'], marker='o', linestyle='-')
plt.title('Time Series of Comomentum Measure (Winners & Losers Deciles)')
plt.xlabel('Date')
plt.ylabel('Average Abnormal Residual Correlation')
plt.show()

# %%
