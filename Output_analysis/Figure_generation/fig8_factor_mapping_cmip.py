import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy.ma as ma
import sys
plt.switch_backend('agg')
plt.ioff()

#design = str(sys.argv[1])

LHsamples = np.loadtxt('../../Qgen/' + design + '.txt')
# convert streamflow multipliers/deltas in LHsamples to HMM parameter values
def convertMultToParams(multipliers):
    params = np.zeros(np.shape(multipliers))
    params[:,0] = multipliers[:,0]*15.258112 # historical dry state mean
    params[:,1] = multipliers[:,1]*0.259061 # historical dry state std dev
    params[:,2] = multipliers[:,2]*15.661007 # historical wet state mean
    params[:,3] = multipliers[:,3]*0.252174 # historical wet state std dev
    params[:,4] = multipliers[:,4] + 0.679107 # historical dry-dry transition prob
    params[:,5] = multipliers[:,5] + 0.649169 # historical wet-wet transition prob
    
    return params

# convert HMM parameter values to streamflow multipliers/deltas in LHsamples 
def convertParamsToMult(params):
    multipliers = np.zeros(np.shape(params))
    multipliers[:,0] = params[:,0]/15.258112 # historical dry state mean
    multipliers[:,1] = params[:,1]/0.259061 # historical dry state std dev
    multipliers[:,2] = params[:,2]/15.661007 # historical wet state mean
    multipliers[:,3] = params[:,3]/0.252174 # historical wet state std dev
    multipliers[:,4] = params[:,4] - 0.679107 # historical dry-dry transition prob
    multipliers[:,5] = params[:,5] - 0.649169 # historical wet-wet transition prob
    
    return multipliers

# find SOWs where mu0 > mu1 and swap their wet and dry state parameters
HMMparams = convertMultToParams(LHsamples[:,[7,8,9,10,11,12]])
for i in range(np.shape(HMMparams)[0]):
    if HMMparams[i,0] > HMMparams[i,2]: # dry state mean above wet state mean
        # swap dry and wet state parameters to correct labels
        mu0 = HMMparams[i,2]
        std0 = HMMparams[i,3]
        mu1 = HMMparams[i,0]
        std1 = HMMparams[i,1]
        p00 = HMMparams[i,5]
        p11 = HMMparams[i,4]
        newParams = np.array([mu0, std0, mu1, std1, p00, p11])
        LHsamples[i,[7,8,9,10,11,12]] = convertParamsToMult(newParams)
        
CMIPsamples = np.loadtxt('../../Qgen/CMIP_SOWs.txt') 
realizations = 10
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'Shoshone','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift']

all_IDs = ['7000550','7200799','3704614']
nStructures = len(all_IDs)

# deal with fact that calling result.summary() in statsmodels.api
# calls scipy.stats.chisqprob, which no longer exists
scipy.stats.chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)

n=12 # Number of months in a year for reshaping data
nMonths = n * 105 #Record is 105 years long

years = np.arange(1909, 2014)
years_s = np.arange(1950, 2014)

# Load historic data
historic_data = np.load('../../Summary_info/historic_flows.npy')*1233.4818

# Load CMIP flows
CMIP3_flows = np.genfromtxt('../../Summary_info/CMIP3_flows.csv', delimiter=',')*1233.4818
CMIP3_flows = np.reshape(CMIP3_flows, [112, 64, 12])
CMIP5_flows = np.genfromtxt('../../Summary_info/CMIP5_flows.csv', delimiter=',')*1233.4818
CMIP5_flows = np.reshape(CMIP5_flows, [97, 64, 12])
baseCase = np.load('../../Summary_info/stationarysynthetic_flows.npy')*1233.4818
synthetic_flows = np.load('../../Summary_info/'+design+'_flows.npy')*1233.4818

colors = ['#DD7373', '#305252', '#3C787E','#D0CD94', '#9597a3', 'red'] #'#AA1209'
labels=['Historic', 'Stationary synthetic', 'CMIP3', 'CMIP5', 'This experiment', 'Failure'] #'Paleo'

# Set arrays for shortage frequencies and magnitudes
frequencies = np.arange(10, 110, 10)
magnitudes = np.arange(10, 110, 10)

def plotfailureheatmap(ID):                 
    allSOWs = np.load('./'+ ID + '_heatmap.npy')
    return(allSOWs)       
'''
Plot LR contours and CMIP points
'''
fig, axes = plt.subplots(3,2, figsize=(18,12))
freq = [7,1,7,0,2,0]
mag = [0,7,0,3,6,7]
heatmaps = [plotfailureheatmap(all_IDs[i])/100 for i in range(len(all_IDs))]
for i in range(len(axes.flat)):
    ax = axes.flat[i]
    allSOWsperformance = heatmaps[int(i/2)]
    success=allSOWsperformance[freq[i],mag[i],:]
    mask = success<1
    failure_flows = synthetic_flows.copy()
    failure_flows = failure_flows[mask]
    data = [np.sum(historic_data, axis=1), np.sum(baseCase, axis=2), np.sum(CMIP3_flows, axis=2), np.sum(CMIP5_flows, axis=2), np.sum(synthetic_flows, axis=2), np.sum(failure_flows, axis=2)] 
    violinplots=ax.violinplot(data, vert=True)
    violinplots['cbars'].set_edgecolor('black')
    violinplots['cmins'].set_edgecolor('black')
    violinplots['cmaxes'].set_edgecolor('black')
    for i in range(len(violinplots['bodies'])):
        vp = violinplots['bodies'][i]
        vp.set_facecolor(colors[i])
        vp.set_edgecolor('black')
        vp.set_alpha(1)
    ax.set_yscale( "log" )
    ax.set_ylabel('Flow at Last Node ($m^3$)',fontsize=20)
    ax.set_xticks(np.arange(1,6))
    ax.set_xticklabels(labels,fontsize=16)    

fig.tight_layout()
fig.savefig('./Paper1_figures/Figure_8_'+design+'_streamflows.svg')
plt.close()


