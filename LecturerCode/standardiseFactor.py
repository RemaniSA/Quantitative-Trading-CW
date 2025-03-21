import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def standardiseFactor(factorRaw):

    # Standardises a factor cross-sectionally by first subtracting
    # the cross-sectional mean from the raw factor exposures at each
    # point in time and then dividing this difference by the cross-sectional
    # standard deviation. Standardised factor exposures have a cross-sectional
    # mean of zero and a cross-sectional standard deviation of one.
    #
    # INPUTS: factorRaw = TxN array of raw factor exposures, e.g. B/P ratios (where
    #                     T is the number of time periods and N is the number of assets
    #
    # OUTPUTS: factorStd = TxN array of standardised factor exposures

    N = factorRaw.shape[1]

    # standardise factor (subtract mean from raw factor exposure and divide this difference
    # by std. dev. cross-sectionally)

    avg = np.tile(np.nanmean(factorRaw,axis=1,keepdims=True),(1,N))
    stdev = np.tile(np.nanstd(factorRaw,axis=1,ddof=1,keepdims=True),(1,N))

    return (factorRaw - avg) / stdev
