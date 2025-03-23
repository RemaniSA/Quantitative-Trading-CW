# solveCAPMExercise.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCAPMStats(r, rIdx):

    # This function computes CAPM betas ("beta"), estimated systematic (i.e. market)
    # risk ("sysRisk") and company-specific risk ("specRisk")
    #
    # INPUTS: r    = TxN array of stock returns (in columns; T = number of periods, N = number of stocks)
    #         rIdx = Tx1 vector of market returns
    #
    # OUTPUTS: beta     = 1xN vector of estimated beta coefficients
    #          sysRisk  = 1xN vector of stocks' systematic risk estimates
    #          specRisk = 1xN vector of stocks' company-specific risk estimates

    [T,N] = r.shape

    # initialise variables
    beta = np.full(N, np.nan)
    specRisk = np.full(N, np.nan)

    for i in np.arange(0,N):

        X = np.hstack((np.ones((T,1)), rIdx))
        Y = r[:,i]

        # run time series regressions for each stock
        betas = np.linalg.lstsq(X, Y, rcond=None)[0]

        beta[i] = betas[1]
        specRisk[i] = np.nanstd(Y - X.dot(betas))


    sysRisk = beta * np.nanstd(rIdx)

    return beta, sysRisk, specRisk


datadir = 'XXX'

r = pd.read_excel(datadir+'stockReturns.xlsx',sheet_name='Data',usecols=[1,2,3,4])
rIdx = pd.read_excel(datadir+'stockReturns.xlsx',sheet_name='Data',usecols=[5])
names = r.columns.tolist()

beta, sysRisk, specRisk = computeCAPMStats(r.values, rIdx.values)

plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.bar(names,beta)
plt.title('Betas')

plt.subplot(3,1,2)
plt.bar(names,sysRisk)
plt.title('Systematic Risk')

plt.subplot(3,1,3)
plt.bar(names,specRisk)
plt.title('Specific Risk')
plt.show()

