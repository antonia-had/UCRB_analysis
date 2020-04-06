import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

def readFiles(filename, firstLine, numSites):
    # read in all monthly flows and re-organize into nyears x 12 x nsites matrix
    with open(filename,'r') as f:
        all_split_data = [x.split('.') for x in f.readlines()]
        
    f.close()
    
    numYears = int((len(all_split_data)-firstLine)/numSites)
    MonthlyQ = np.zeros([12*numYears,numSites])
    sites = []
    for i in range(numYears):
        for j in range(numSites):
            index = firstLine + i*numSites + j
            sites.append(all_split_data[index][0].split()[1])
            all_split_data[index][0] = all_split_data[index][0].split()[2]
            MonthlyQ[i*12:(i+1)*12,j] = np.asfarray(all_split_data[index][0:12], float)
            
    MonthlyQ = np.reshape(MonthlyQ,[int(np.shape(MonthlyQ)[0]/12),12,numSites])
            
    return MonthlyQ

def revertCumSum(cumulative):
    '''Revert cumulative sum. Modified from https://codereview.stackexchange.com/questions/117183/extracting-original-values-from-cumulative-sum-values'''
    output = [0] * len(cumulative)
    for i,e in reversed(list(enumerate(cumulative))):
        output[i]=cumulative[i] - cumulative[i-1]
    output[0]=cumulative[0]

    return output

# read in monthly flows at all sites
MonthlyQ = readFiles('../cm2015x.xbm', 16, 208)

# calculate fraction of annual flow received each month under shifts of 1-60 days
LastNodeFractions = np.zeros([2013-1951,61,12])
for i in range(2013-1951):
    LastNodeFractions[i,0,:] = MonthlyQ[43+i,:,-1]/np.sum(MonthlyQ[43+i,:,-1])

# read in daily flows at last node
LastNodeQ = pd.read_csv('../CO_River_UT_State_line.csv')
LastNodeQ['Date'] = pd.to_datetime(LastNodeQ['Date'],format="%Y-%m-%d")
LastNodeQ['Year'] = LastNodeQ['Date'].dt.year
LastNodeQ['Month'] = LastNodeQ['Date'].dt.month
# increase year by 1 for Oct->Dec to conver to water year
indices = np.where(LastNodeQ['Month'] >= 10)[0]
LastNodeQ['Year'][indices] += 1
    
# create column of dataframe for shifted flows
LastNodeQ['ShiftedFlow'] = LastNodeQ['Flow']

years = np.unique(LastNodeQ['Year'])
shifts = range(1,61)
for shift in shifts:
    LastNodeQ['ShiftedFlow'][0:-shift] = LastNodeQ['Flow'][shift::]
    MonthlyTotals = LastNodeQ.set_index('Date').resample('M').sum()
    MonthlyTotals['Year'] = MonthlyTotals.index.year
    MonthlyTotals['Month'] = MonthlyTotals.index.month
    # reduce year by 1 for Jan->Sept to conver to water year
    indices = np.where(MonthlyTotals['Month'] >= 10)[0]
    MonthlyTotals['Year'][indices] += 1
    # convert Monthly totals from cfs to acre-ft
    MonthlyTotals['Flow'] = MonthlyTotals['Flow'] * 2.29569E-05 * 86400
    MonthlyTotals['ShiftedFlow'] = MonthlyTotals['ShiftedFlow'] * 2.29569E-05 * 86400
    
    for i in range(len(years)-1):
        year = years[i]
        flows = np.where(MonthlyTotals['Year']==year)[0]
        
        # calculate cumulative flows at gage w/ and w/o the shift, and of the naturalized flows
        gage_cdf = np.cumsum(MonthlyTotals['Flow'][flows])
        gage_shifted_cdf = np.cumsum(MonthlyTotals['ShiftedFlow'][flows])
        natural_cdf = np.cumsum(MonthlyQ[43+i,:,-1])
        
        # normalize cdfs to sum to 1
        gage_cdf = gage_cdf/np.max(gage_cdf)
        gage_shifted_cdf = gage_shifted_cdf/np.max(gage_shifted_cdf)
        natural_cdf = natural_cdf/np.max(natural_cdf)
        
        # apply same shift to natural flows as at gage
        natural_shifted_cdf = natural_cdf + gage_shifted_cdf - gage_cdf
        
        # compute monthly fractional contribution
        LastNodeFractions[i,shift,:] = revertCumSum(natural_shifted_cdf)

        if i == 33 and shift == 60: # plot 60-day cdf shift for 1985
            sns.set_style("darkgrid")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            l1, = ax.step(range(12),gage_cdf,linewidth=2,c='b')
            l2, = ax.step(range(12),gage_shifted_cdf,linewidth=2,linestyle='--',c='b')
            l3, = ax.step(range(12),natural_cdf,linewidth=2,c='g')
            l4, = ax.step(range(12),natural_shifted_cdf,linewidth=2,linestyle='--',c='g')
            ax.set_xlabel('Month',fontsize=16)
            ax.set_ylabel('Cumulative Fraction of Annual Flow',fontsize=16)
            ax.tick_params(axis='both',labelsize=14)
            ax.legend([l1,l2,l3,l4],['Gage','Gage Shifted','Natural','Natural Shifted'],loc='lower right',fontsize=14)
            fig.savefig('FigureS4a.pdf')
            fig.clf()

MonthlyQ = MonthlyQ*1233.48/1E6 # convert Monthly flows to millions of m^3

# for 1985, make a plot of the base and shifted hydrographs
i = 33
cmap = matplotlib.cm.get_cmap('coolwarm')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(MonthlyQ[43+i,:,-1])
for shift in shifts:
    ax.plot(np.sum(MonthlyQ[43+i,:,-1]) * LastNodeFractions[i,shift,:], c=cmap(shift/61))
    
ax.set_xticks(range(12))
ax.set_xticklabels(['O','N','D','J','F','M','A','M','J','J','A','S'])
ax.set_xlim([0,11])
ax.tick_params(axis='both',labelsize=14)
ax.set_ylabel('Monthly flow (millions of m^3)',fontsize=16)
ax.set_title('WY ' + str(1952+i),fontsize=16)
sm = matplotlib.cm.ScalarMappable(cmap=cmap)
sm.set_array([0,60])
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_ylabel('Shift in Peak (days earlier)',fontsize=16)
fig.subplots_adjust(right=0.8,left=0.15)
fig.set_size_inches([8.4,4.8])
fig.savefig('FigureS4b.pdf')
fig.clf()