# solveFamaMacBethExercise.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def famaMacBeth(factor, returns):

    [T, N] = factor.shape

    gamma = np.full(T, np.nan)

    for t in np.arange(1,T):

        Y = returns[t,np.newaxis].T
        X = np.hstack((np.ones((N,1)), factor[t-1,np.newaxis].T))

        gamma[t] = np.linalg.lstsq(X, Y, rcond=None)[0][1]


    tstat = np.nanmean(gamma) / (np.nanstd(gamma) / np.sqrt(T))

    return gamma, tstat



datadir = 'XXX'

dates = pd.read_excel(datadir+'uk_data.xlsx',sheet_name='returns',usecols=[0])
r = pd.read_excel(datadir+'uk_data.xlsx',sheet_name='returns',usecols=[1,2,3,4,5])
betas = pd.read_excel(datadir+'uk_data.xlsx',sheet_name='beta',usecols=[1,2,3,4,5])
names = r.columns.tolist()

[gamma, tstat] = famaMacBeth(betas.values, r.values)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(gamma)
plt.title('Factor Returns')

plt.subplot(2,1,2)
plt.bar(['Mean Factor Return','T-Stat'],[np.nanmean(gamma),tstat])
plt.title('Mean Factor Return & T-Stat')
plt.show()

