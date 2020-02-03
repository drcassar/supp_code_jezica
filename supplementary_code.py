import pandas as pd
import numpy as np
from numpy import log10
from numpy.random import normal
from scipy.stats import spearmanr, linregress
from scipy.constants import pi

np.random.seed(42)

numberOfDraws = 100000

# JZCA Tliquidus factor
JZCATliqfaclist = [0.94, 1]

# percentile confidence
percentile = 95

# shape factor
g = pi

# base values for Ns and Xs
baseLogNs = 3
baseLogXs = -2

# standard deviation for the noise
stdTliq = 2
stdUmax = 2
stdTUmax = 2
stdLogNs = 1

for JZCATliqfac in JZCATliqfaclist:

    # load data table
    data = pd.read_csv(r'./data.csv')

    # get experimental data
    expTliq = data['Tliq'].values
    expUmax = data['Umax'].values
    expTUmax = data['TUmax'].values

    # reshape for multiplication
    expTliq = expTliq.reshape((len(expTliq), 1))
    expUmax = expUmax.reshape((len(expUmax), 1))
    expTUmax = expTUmax.reshape((len(expTUmax), 1))

    # generate noise
    drawShape = (numberOfDraws, len(data))
    noiseTliq = 1 + normal(0, stdTliq, drawShape) / 100
    noiseUmax = 1 + normal(0, stdUmax, drawShape) / 100
    noiseTUmax = 1 + normal(0, stdTUmax, drawShape) / 100

    # get the distribution with noise
    distTliq = expTliq * noiseTliq.T
    distUmax = expUmax * noiseUmax.T
    distTUmax = expTUmax * noiseTUmax.T
    distTnCalc = JZCATliqfac * distTliq

    # When TUmax > Tliq
    logic = distTUmax > distTliq
    distTUmax[logic] = distTliq[logic] - 1

    # get the distribution of Ns
    distNs = 10**(normal(baseLogNs, stdLogNs, numberOfDraws))
    distXs = 10**baseLogXs

    # viscosity
    ninf = data['VFT_ninf'].values
    A = data['VFT_B'].values
    T0 = data['VFT_T0'].values

    # reshape for multiplication
    T0 = T0.reshape((len(T0), 1))
    A = A.reshape((len(A), 1))
    ninf = ninf.reshape((len(ninf), 1))

    # JZCA
    distLogViscosity = ninf + A / (distTnCalc - T0)
    distViscosity = 10**distLogViscosity
    distJzca = distViscosity / distTliq**2
    medianJzca = np.median(distJzca, axis=1)
    upperJzca = np.percentile(distJzca, percentile, axis=1)
    lowerJzca = np.percentile(distJzca, 100 - percentile, axis=1)

    # Rc calculation
    disttn = (distXs / (g * distNs * distUmax**2))**(1 / 2)
    distRcCalc = (distTliq - distTUmax) / disttn
    medianRcCalc = np.median(distRcCalc, axis=1)
    upperRcCalc = np.percentile(distRcCalc, percentile, axis=1)
    lowerRcCalc = np.percentile(distRcCalc, 100 - percentile, axis=1)

    # Update table
    data['median_JZCA'] = medianJzca
    data['upper_JZCA'] = upperJzca
    data['lower_JZCA'] = lowerJzca
    data['median_Rc_calculated'] = medianRcCalc
    data['upper_Rc_calculated'] = upperRcCalc
    data['lower_Rc_calculated'] = lowerRcCalc

    # Save table
    data.to_csv(rf'./JZCA_error_{JZCATliqfac}.csv', index=False)

    # Spearman
    rho = []
    p = []
    for x, y in zip(distRcCalc.T, distJzca.T):
        logic = np.logical_or(np.isnan(x), np.isnan(y))
        corr, p_value = spearmanr(x[~logic], y[~logic])
        rho.append(corr)
        p.append(p_value)

    rhoDF = pd.DataFrame({'rho': rho, 'p': p})
    rhoDF.to_csv(rf'./rho_{JZCATliqfac}.csv')

    # Linear regression
    r2 = []
    r = []
    plin = []
    slopelin = []
    interlin = []
    for x_, y_ in zip(distRcCalc.T, distJzca.T):
        x, y = log10(x_), log10(y_)
        logic = np.logical_or(np.isnan(x), np.isnan(y))
        slope, intercept, r_value, p_value, std_err = linregress(
            x[~logic], y[~logic])
        r.append(r_value)
        plin.append(p_value)
        slopelin.append(slope)
        interlin.append(intercept)

    r2 = np.array(r)**2

    r2DF = pd.DataFrame({
        'r': r,
        'p': plin,
        'slope': slopelin,
        'intercept': interlin,
        'r2': r2
    })
    r2DF.to_csv(rf'./r2_{JZCATliqfac}.csv')
