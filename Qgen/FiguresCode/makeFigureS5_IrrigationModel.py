import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats
import seaborn as sns

def norm_MC(Nyears,theoretical,dataCorr):
    corr = np.zeros([10000])
    for i in range(10000): # 10,000 MC simulations
        simulated = stats.norm.rvs(0,1, size=Nyears)
        corr[i] = np.corrcoef(np.sort(simulated),theoretical)[0,1]
 
    # find significance levels
    corr = np.sort(corr)
    for i in range(10000):
        if dataCorr > corr[i]:
            sigLevel = (i+1)/10000.0
 
    return sigLevel

AnnualIWR = np.loadtxt('../AnnualIWR.csv',delimiter=',')*1233.48 # convert to m^3
AnnualQ = np.array(pd.read_csv('../AnnualQ.csv'))

IWRsums = np.sum(AnnualIWR,1)
Qsums = AnnualQ[:,-1]*1233.48 # convert to m^3

Qsums_prime = Qsums - np.mean(Qsums)
IWRsums_prime = IWRsums - np.mean(IWRsums)

X = np.reshape(Qsums_prime,[len(Qsums_prime),1])
X = X.astype('float')
y = IWRsums_prime
model = sm.OLS(y,X).fit()

sns.set()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Qsums_prime,IWRsums_prime)
xmin = np.min(Qsums_prime)
xmax = np.max(Qsums_prime)
ax.plot([xmin,xmax],[model.params[0]*xmin,model.params[0]*xmax],c='r')
ax.set_xlabel('Annual Flow Anomaly at CO-UT State Line (m^3)',fontsize=16)
ax.set_ylabel('Annual Total Irrigation Demand Anomaly (m^3)',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
fig.set_size_inches([7.33,5.66])
fig.savefig('FigureS5a.pdf')
fig.clf()

# make sure residuals uncorrelated in time
fig, axes = plt.subplots(1,1)
ax = plt.subplot(111)
fig = sm.graphics.tsa.plot_acf(model.resid, ax=ax)
ax.set_xlabel('Number of Lags',fontsize=16)
ax.set_xlim([0,24])
ax.set_ylabel('Correlation of Residuals',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
fig.set_size_inches([7.67,5.75])
fig.savefig('FigureS5b.pdf')
fig.clf()

# test if residuals are normally distributed
Nyears = np.shape(AnnualQ)[0]
m = np.arange(Nyears)+1
p = (m-0.5)/Nyears
norm = stats.norm.ppf(p,0,1)
normRho = np.corrcoef((np.sort(model.resid)-np.mean(model.resid))/np.std(model.resid),norm)[0,1]
normSigLevel = norm_MC(Nyears, norm, normRho)

# make QQ plot of residuals
plt.scatter(norm,(np.sort(model.resid)-np.mean(model.resid))/np.std(model.resid))
plt.plot([-3,3],[-3,3],c='r')
plt.xlabel('Theoretical Quantiles',fontsize=16)
plt.ylabel('Residual Quantiles',fontsize=16)
plt.tick_params(axis='both',labelsize=14)
plt.savefig('FigureS5c.pdf')
plt.clf()

# make sure residuals homoskedastic
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(model.fittedvalues,model.resid)
ax.set_xlabel('Fitted Values',fontsize=16)
ax.set_ylabel('Residuals',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
fig.set_size_inches([7.67,5.75])
fig.savefig('FigureS5d.pdf')
fig.clf()