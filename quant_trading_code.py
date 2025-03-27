#%%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helper_functions.standardiseFactor import standardiseFactor

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice.")

#%%
# Data Loading

# Set directory and file paths
base_dir = os.path.dirname(__file__)
returns_path = os.path.join(base_dir, 'datasets', 'US_Returns.csv')
live_path  = os.path.join(base_dir, 'datasets', 'US_live.csv')
dates_path = os.path.join(base_dir, 'datasets', 'US_Dates.xlsx')
names_path = os.path.join(base_dir, 'datasets', 'US_Names.xlsx')
factors_path = os.path.join(base_dir, 'datasets', 'FamaFrench.csv')

# Load data
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

    # List to store gamma coefficients
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
        
        # Append gamma to the list
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
df_gamma, tstat = famaMacBeth(df_momentum, df_returns, df_live,min_obs=0)
print("Fama–MacBeth Regression - Weekly Momentum Factor Coefficients (Gamma):")
print(df_gamma.head())
print("\nT-Statistic:", tstat)

# Save gamma coefficients to a CSV file
df_gamma.to_csv('datasets/US_FMB_Gamma.csv')

# Compute the mean factor return
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

# List to store valid observation counts per week
valid_counts = []

# Loop over each week in the momentum factor DataFrame
for t in df_momentum.index:
    # Count valid observations (live stock with non-missing momentum)
    valid = (df_live.loc[t] == 1) & (df_momentum.loc[t].notna())
    valid_counts.append(valid.sum())

valid_counts_series = pd.Series(valid_counts, index=df_momentum.index)

# Print summary statistics for valid observation counts
print("Summary of Valid Observation Counts per Week:")
print(valid_counts_series.describe())
print("\nMedian valid observations:", valid_counts_series.median())
print("\nQuantiles (25%, 50%, 75%):")
print(valid_counts_series.quantile([0.25, 0.5, 0.75]))

# Create grid of minimum observation thresholds to test
# We start from 800 and increment by 100 up to 2000
grid_min_obs = np.arange(800, 2000, 100)
first_significant = None

# Loop over each threshold in the grid
for min_obs in grid_min_obs:
    # Run Fama-MacBeth regression with the current threshold
    df_gamma_temp, tstat_temp = famaMacBeth(df_momentum, df_returns, df_live, min_obs=min_obs)
    print(f"Minimum obs = {min_obs}: t-statistic = {tstat_temp:.4f}")
    # Check if t-statistic is significant
    if np.abs(tstat_temp) >= 1.96:
        first_significant = min_obs
        # Print the first significant t-statistic found
        print(f"First significant t-stat found with min_obs = {min_obs}, t-statistic = {tstat_temp:.4f}")
        break

# Fall-back message if no significant t-statistic is found
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
    # Standardise momentum cross-sectionally (for all dates at once)
    df_momentum_std = pd.DataFrame(
        standardiseFactor(df_momentum.values),
        index=df_momentum.index,
        columns=df_momentum.columns
    )

    # Extract momentum values for the current date
    current_mom = df_momentum_std.loc[current_date]
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
        """
        Calculate the average off-diagonal correlation for a given DataFrame group.
        This function computes the average of all off-diagonal elements in the 
        correlation matrix of the input DataFrame. The off-diagonal elements 
        represent the pairwise correlations between different columns.
        Parameters:
        -----------
        df_group : pandas.DataFrame
            A DataFrame containing numerical data for which the average 
            off-diagonal correlation is to be calculated.
        Returns:
        --------
        float
            The average off-diagonal correlation. Returns NaN if the DataFrame 
            has fewer than 2 columns.
        Notes:
        ------
        - If the input DataFrame has fewer than 2 columns, the function returns NaN 
          since a correlation matrix cannot be computed.
        """
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

# Display the computed comomentum measure
print("Comomentum Measure (first few rows):")
print(df_comomentum.head())

# Save the comomentum measure to a CSV file
df_comomentum.to_csv('datasets/US_Comomentum.csv')

# Plot the comomentum measure over time
plt.figure(figsize=(10,6))
plt.plot(df_comomentum.index, df_comomentum['Comomentum'], marker='o', linestyle='-')
plt.title('Time Series of Comomentum Measure (Winners & Losers Deciles)')
plt.xlabel('Date')
plt.ylabel('Average Abnormal Residual Correlation')
plt.show()

# %%
# Check the correlation between mom and comom:

# Compute row-wise means
mom_mean = df_momentum.mean(1)
comom_mean = df_comomentum.mean(1)

# Combine into a single DataFrame with named columns and drop NaNs
df_corr = pd.concat([mom_mean, comom_mean], axis=1)
df_corr.columns = ['momentum', 'comomentum']
df_corr = df_corr.dropna(how='all')

# Compute correlation
correlation = df_corr['momentum'].corr(df_corr['comomentum'])

# Print the correlation
print(f'Correlation between Momentum and Comomentum: {correlation:.4f}')


# %%
# Q5+6: Adjust Momentum Factor Using Comomentum
#
# We implement two approaches to adjust momentum:
#
# Approach 1: Continuous Weight Adjustment
#   Adjust momentum by:
#       M_adj_cont = M * f_cont,  where f_cont = 1 / (1 + lambda * (C_t - C_bar))
#
# Approach 2: Threshold-Based Adjustment
#   Adjust momentum by:
#       M_adj_thresh = M * I{ C_t <= T }
#
# M is the momentum factor (df_momentum), C_t is comomentum (df_comomentum).
# C_bar is median of comomentum and T is a threshold (e.g., 75th percentile).
#
# Dummy Parameters (need to optimise via grid search or cross-validation):
lambda_val = 5.0
C_bar = df_comomentum['Comomentum'].median()
threshold_val = df_comomentum['Comomentum'].quantile(0.795)

# Create copies of df_momentum for the adjusted factors
df_momentum_adj_cont = df_momentum.copy()
df_momentum_adj_thresh = df_momentum.copy()

# Loop over each date in df_momentum and apply the adjustments
for date in df_momentum.index:
    # Ensure comomentum is available for this date
    if date not in df_comomentum.index:
        continue
    C_t = df_comomentum.loc[date, 'Comomentum']
    # Continuous adjustment factor: higher comomentum reduces the weight
    f_cont = 1.0 / (1.0 + lambda_val * (C_t - C_bar))
    # Threshold adjustment factor: 1 if comomentum is below threshold, 0 otherwise
    f_thresh = 1.0 if C_t <= threshold_val else 0.0
    # Multiply entire row (i.e., for all stocks) by adjustment factor
    df_momentum_adj_cont.loc[date] = df_momentum.loc[date] * f_cont
    df_momentum_adj_thresh.loc[date] = df_momentum.loc[date] * f_thresh

# For each method, aggregate across stocks to obtain a time series
standard_mom_avg = df_momentum.mean(1)
adj_cont_mom_avg = df_momentum_adj_cont.mean(1)
adj_thresh_mom_avg = df_momentum_adj_thresh.mean(1)

# Combine these series into one DataFrame
df_adjusted = pd.DataFrame({
    'Standard_Momentum': standard_mom_avg,
    'Adjusted_Continuous': adj_cont_mom_avg,
    'Adjusted_Threshold': adj_thresh_mom_avg
})

# Compute summary statistics for each series
summary_stats = df_adjusted.describe()
print("Summary Statistics for Momentum Factors:")
print(summary_stats)

# Annualised mean and standard deviation
annualised_mean = summary_stats.loc['mean'] * 52
annualised_std = summary_stats.loc['std'] * np.sqrt(52)

# Print annualised mean and standard deviation
print("\nAnnualised Mean:")
print(annualised_mean)
print("\nAnnualised Standard Deviation:")
print(annualised_std)

# Compute cumulative returns
df_cum_returns = df_adjusted.cumsum()
print("Cumulative Returns (first few rows):")
print(df_cum_returns.head())

# Plot cumulative returns using seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(12,8))
sns.lineplot(data=df_cum_returns)
plt.title("Cumulative Returns of Adjusted Momentum Factors")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()

# %%

# Sharpe Ratio Over Time

df_cum_sharpe = pd.DataFrame(index=df_adjusted.index, columns=df_adjusted.columns)

# Loop over each adj. momentum factor column
for col in df_adjusted.columns:
    # Compute excess returns
    excess_returns = df_adjusted[col] - df_factors["RF"]
    
    # Compute the cumulative mean and standard deviation
    cum_mean = excess_returns.expanding().mean()
    cum_std = excess_returns.expanding().std()
    
    # Calculate the cumulative Sharpe ratio
    df_cum_sharpe[col] = cum_mean / cum_std * np.sqrt(52) # annualised

# Now df_cum_sharpe holds the cumulative Sharpe ratio up to each date for each column
df_cum_sharpe = df_cum_sharpe.dropna()
print(df_cum_sharpe)

# Plot the cumulative Sharpe ratios
plt.figure(figsize=(12,8))
sns.lineplot(data=df_cum_sharpe[df_cum_sharpe.index >= '1996-01-01'])
plt.title("Cumulative Sharpe Ratios of Adjusted Momentum Factors")
plt.xlabel("Date")
plt.ylabel("Cumulative Sharpe Ratio")
plt.legend()
plt.show()

# %%
# Q5+6 part 2: Parameter Tuning via Walk-Forward (Rolling Window) Grid Search

# We collect both the best parameters and the performance metric 
# for each iteration and for each parameter set.
# We then review the results to determine the best parameter set.

# Define parameter grids: 
lambda_grid = np.concatenate([    # candidate values for continuous adjustment parameter
    np.linspace(0.1, 10, 10),     # fine grid for small lambda
    np.linspace(15, 50, 8),       # coarser grid for moderate lambda
    np.linspace(60, 100, 5)       # sparse grid for large lambda
])

threshold_grid = np.linspace(0.795, 0.815, 9)  # candidate quantile thresholds for threshold adjustment

# Define performance metric (here, cumulative return)
def performance_metric(series):
    return series.sum()

# Set window sizes (in weeks)
train_window = 100   # training period length
test_window = 10     # validation period length

# List to store tuning results for each walk-forward iteration
tuning_results = []

# Determine start and end positions for the rolling windows
start_pos = train_window - 1
end_pos = len(df_returns.index) - test_window

# Loop over each window in steps of test_window
for pos in range(start_pos, end_pos, test_window):
    # Define training and testing periods
    train_dates = df_returns.index[pos - train_window + 1 : pos + 1]
    test_dates = df_returns.index[pos + 1 : pos + test_window + 1]
    
    # Ensure we only use dates that exist in df_comomentum:
    train_dates = train_dates.intersection(df_comomentum.index)
    test_dates = test_dates.intersection(df_comomentum.index)
    if len(train_dates) == 0 or len(test_dates) == 0:
        continue
    
    # Subset momentum and comomentum for the training period
    train_mom = df_momentum.loc[train_dates]
    train_comom = df_comomentum.loc[train_dates]
    
    # Subset returns for the test period (if needed for other evaluations)
    test_returns = df_returns.loc[test_dates]
    
    # --- Continuous Adjustment Grid Search ---
    best_metric_cont = -np.inf
    best_params_cont = None
    for lam in lambda_grid:
        # Compute benchmark (median) comomentum over the training period
        C_bar = train_comom['Comomentum'].median()
        # Adjust momentum in training period using continuous function:
        # f(C_t) = 1 / (1 + lam*(C_t - C_bar))
        adjusted_mom_train = train_mom.copy()
        for date in train_mom.index:
            if date not in train_comom.index:
                continue
            C_t = train_comom.loc[date, 'Comomentum']
            f_cont = 1.0 / (1.0 + lam * (C_t - C_bar))
            adjusted_mom_train.loc[date] = train_mom.loc[date] * f_cont
        train_signal = adjusted_mom_train.mean(axis=1)
        
        # Apply same adjustment for test period using the same benchmark C_bar
        adjusted_mom_test = df_momentum.loc[test_dates].copy()
        for date in test_dates:
            if date not in df_comomentum.index:
                continue
            C_t = df_comomentum.loc[date, 'Comomentum']
            f_cont = 1.0 / (1.0 + lam * (C_t - C_bar))
            adjusted_mom_test.loc[date] = df_momentum.loc[date] * f_cont
        test_signal = adjusted_mom_test.mean(axis=1)
        
        metric_value = performance_metric(test_signal)
        if metric_value > best_metric_cont:
            best_metric_cont = metric_value
            best_params_cont = lam
    
    # --- Threshold-Based Adjustment Grid Search ---
    best_metric_thresh = -np.inf
    best_params_thresh = None
    for thresh in threshold_grid:
        # Use the thresh quantile of comomentum over the training period as the threshold value
        threshold_value = train_comom['Comomentum'].quantile(thresh)
        adjusted_mom_train_thresh = train_mom.copy()
        for date in train_mom.index:
            if date not in train_comom.index:
                continue
            C_t = train_comom.loc[date, 'Comomentum']
            f_thresh = 1.0 if C_t <= threshold_value else 0.0
            adjusted_mom_train_thresh.loc[date] = train_mom.loc[date] * f_thresh
        train_signal_thresh = adjusted_mom_train_thresh.mean(axis=1)
        
        adjusted_mom_test_thresh = df_momentum.loc[test_dates].copy()
        for date in test_dates:
            if date not in df_comomentum.index:
                continue
            C_t = df_comomentum.loc[date, 'Comomentum']
            f_thresh = 1.0 if C_t <= threshold_value else 0.0
            adjusted_mom_test_thresh.loc[date] = df_momentum.loc[date] * f_thresh
        test_signal_thresh = adjusted_mom_test_thresh.mean(axis=1)
        
        metric_value_thresh = performance_metric(test_signal_thresh)
        if metric_value_thresh > best_metric_thresh:
            best_metric_thresh = metric_value_thresh
            best_params_thresh = thresh
    
    # Append both results for the current walk-forward iteration:
    tuning_results.append({
        'train_start': train_dates[0],
        'train_end': train_dates[-1],
        'test_start': test_dates[0],
        'test_end': test_dates[-1],
        'best_params_cont': best_params_cont,
        'performance_cont': best_metric_cont,
        'best_params_thresh': best_params_thresh,
        'performance_thresh': best_metric_thresh
    })

# Convert tuning results to a DataFrame for review
df_tuning_results = pd.DataFrame(tuning_results)
print("Tuning Results (each row corresponds to one walk-forward iteration):")
print(df_tuning_results)

df_tuning_results.to_csv('datasets/US_Momentum_Adjustment_Tuning.csv')

# Get 3 most occuring lambda and their average performance
top_lambdas = (
    df_tuning_results.groupby('best_params_cont')['performance_cont']
    .agg(lambda_count='count', lambda_avg_return='mean')
    .reset_index()
    .sort_values(by='lambda_count', ascending=False)
    .head(3)
)

# Get 3 most occuring thresholds and their average performance
top_thresholds = (
    df_tuning_results.groupby('best_params_thresh')['performance_thresh']
    .agg(threshold_count='count', threshold_avg_return='mean')
    .reset_index()
    .sort_values(by='threshold_count', ascending=False)
    .head(3)
)

print("\nTop 3 Lambda Values (Most Frequently Selected):")
print(top_lambdas)

print("\nTop 3 Threshold Values (Most Frequently Selected):")
print(top_thresholds)

# %%
# Hypothesis Test: Is comomentum actually adding signal?

# We will perform a hypothesis test to determine if the comomentum measure is adding signal to the momentum factor.
# Specifically, we will compare the performance of the adjusted momentum factor using the true comomentum values
# against the performance of the adjusted momentum factor using randomized (null) comomentum values.

# If the randomised dataset returns the same hyperparameters as the true dataset, then the comomentum measure is not adding signal.
# Conclusion: No signal added by comomentum, since the same hyperparameters are selected for the randomised dataset.

# Shuffle comomentum to destroy any true time structure
df_comomentum_shuffled = df_comomentum.copy()
df_comomentum_shuffled['Comomentum'] = np.random.permutation(df_comomentum_shuffled['Comomentum'].values)

# List to store null results
null_results = []

# Loop over each window in steps of test_window
for pos in range(start_pos, end_pos, test_window):
    # Define training and testing periods
    train_dates = df_returns.index[pos - train_window + 1 : pos + 1]
    test_dates = df_returns.index[pos + 1 : pos + test_window + 1]

    # Ensure we only use dates that exist in df_comomentum:
    train_dates = train_dates.intersection(df_comomentum.index)
    test_dates = test_dates.intersection(df_comomentum.index)
    if len(train_dates) == 0 or len(test_dates) == 0:
        continue

    # Subset momentum and comomentum for the training period
    train_mom = df_momentum.loc[train_dates]
    train_comom = df_comomentum_shuffled.loc[train_dates]
    # Subset returns for the test period
    test_returns = df_returns.loc[test_dates]

    # --- Continuous Adjustment Grid Search ---
    # Perform grid search for the best lambda value
    best_metric_cont = -np.inf
    best_params_cont = None
    # Loop over each lambda value in the grid
    for lam in lambda_grid:
        # Compute benchmark (median) comomentum over the training
        C_bar = train_comom['Comomentum'].median()

        # Adjust momentum in training period using continuous function:   
        adjusted_mom_test = df_momentum.loc[test_dates].copy()
        for date in test_dates:
            if date not in df_comomentum_shuffled.index:
                continue
            C_t = df_comomentum_shuffled.loc[date, 'Comomentum']
            f_cont = 1.0 / (1.0 + lam * (C_t - C_bar))
            adjusted_mom_test.loc[date] = df_momentum.loc[date] * f_cont
        # Compute the signal for the test period
        test_signal = adjusted_mom_test.mean(axis=1)
        # Compute the performance metric
        metric_value = performance_metric(test_signal)
        # Update the best metric and parameters if needed
        if metric_value > best_metric_cont:
            best_metric_cont = metric_value
            best_params_cont = lam

    # Append the results for the current iteration
    null_results.append({
        'test_start': test_dates[0],
        'test_end': test_dates[-1],
        'best_params_cont': best_params_cont,
        'performance_cont': best_metric_cont
    })

# Results for null (randomised) comomentum
df_null_results = pd.DataFrame(null_results)

# Top lambdas under the null
top_null_lambdas = (
    df_null_results.groupby('best_params_cont')['performance_cont']
    .agg(lambda_count='count', lambda_avg_return='mean')
    .reset_index()
    .sort_values(by='lambda_count', ascending=False)
    .head(3)
)

print("\n[H₀] Top 3 Lambda Values with Randomized Comomentum:")
print(top_null_lambdas)

# %%

# Rerun Fama-MacBeth regression using the adjusted continuous momentum factor

# Define the best lambda value
best_lambda = 5

# Adjust momentum using the best lambda value
C_bar = df_comomentum['Comomentum'].median()

# Loop over each date in df_momentum and apply the continuous adjustment
for date in df_momentum.index:
    if date not in df_comomentum.index:
        continue
    C_t = df_comomentum.loc[date, 'Comomentum']
    f_cont = 1.0 / (1.0 + best_lambda * (C_t - C_bar))
    df_momentum_adj_cont.loc[date] = df_momentum.loc[date] * f_cont

# Run Fama-MacBeth regression using the adjusted momentum factor
df_gamma_best, tstat_best = famaMacBeth(df_momentum_adj_cont, df_returns, df_live, min_obs=0)
print("Fama–MacBeth Regression - Weekly Adjusted Momentum Factor Coefficients (Gamma):")
print(df_gamma_best.head())

# Save gamma coefficients to a CSV file
df_gamma_best.to_csv('datasets/US_FMB_Gamma_Adjusted.csv')

mean_factor_return_best = df_gamma_best.mean()
print(f'Mean Factor Return (Adjusted): {mean_factor_return_best[0]:.4f}')

# TStat
print("\nT-Statistic (Adjusted):", tstat_best)
# %%

# Run Fama-MacBeth regression on threshold-adjusted momentum factor
df_gamma_thresh, tstat_thresh = famaMacBeth(df_momentum_adj_thresh, df_returns, df_live, min_obs=0)

# Output results
print("Fama–MacBeth Regression - Weekly Threshold-Adjusted Momentum Factor Coefficients (Gamma):")
print(df_gamma_thresh.head())

# Display mean factor return and t-statistic
mean_factor_return_thresh = df_gamma_thresh.mean()
print(f'Mean Factor Return (Threshold-Adjusted): {mean_factor_return_thresh[0]:.4f}')
print(f"T-Statistic (Threshold-Adjusted): {tstat_thresh:.4f}")


# %%
