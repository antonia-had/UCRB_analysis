import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy import stats as ss

def sortFlows(f):
    F = np.empty(np.shape(f))
    F[:] = np.NaN
    for i in range(np.shape(F)[0]):
        F[i,:] = np.sort(f[i,:],0)[::-1]
    
    return F
    
# simulate stationary flows, then vary each parameter one-at-a-time
# to its lower and upper bound
# parameters: IWR, mu0, sigma0, mu1, sigma1, p00, p11
samples = np.ones([1+8*2,8]) # base case for multipliers
samples[:,-2:] = 0 # base case for delta shifts
param_ranges = np.loadtxt('uncertain_params_original.txt',usecols=[1,2])[[0,7,8,9,10,11,12,13],:]
for i in range(np.shape(param_ranges)[0]):
    samples[i*2+1,i] = param_ranges[i,0]
    samples[(i+1)*2,i] = param_ranges[i,1]

# load historical flow data and node locations
MonthlyQ = np.array(pd.read_csv('MonthlyQ.csv'))
AnnualQ = np.array(pd.read_csv('AnnualQ.csv'))
AnnualQ = AnnualQ + 1 # add 1 because site 47 has years with 0 flow

# load CMIP 3 and CMIP 5 flow data at last node
CMIP3flows = np.loadtxt('CMIP3_flows.csv',delimiter=',')
CMIP5flows = np.loadtxt('CMIP5_flows.csv',delimiter=',')

# reshape into nsims x 64 years x 12 months
CMIP3flows = np.reshape(CMIP3flows,[112,64,12]) # site x year x month
CMIP5flows = np.reshape(CMIP5flows,[97,64,12]) # site x year x month

# reshape into [nsims*64] x 12
CMIP3flows = np.reshape(CMIP3flows,[112*64,12])
CMIP5flows = np.reshape(CMIP5flows,[97*64,12])

if not os.path.exists('Figs/GeneratorFigs'):
    os.makedirs('Figs/GeneratorFigs')

# plot historical annual traces at last node and stationary synthetic traces at last node
baseCase = np.load('Sample1_Flows_logspace.npy')[:,-1,:]
fig = plt.figure()
ax = fig.add_subplot(111)
for j in range(np.shape(baseCase)[0]):
    ax.plot(range(np.shape(baseCase)[1]),np.log(baseCase[j,:]),c='#bdbdbd',label='synthetic')
             
for j in range(int(np.shape(MonthlyQ)[0]/12)):
    ax.plot(range(12),np.log(MonthlyQ[j*12:(j+1)*12,-1]),c='k',label='historical')
    
ax.set_xlabel('Month',fontsize=16)
ax.set_ylabel('log(Flow at Last Node)',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
ax.set_xlim([0,11])
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
fig.legend(handles, labels)
fig.suptitle('Stationary Generator')
fig.savefig('Figs/GeneratorFigs/LogStationaryHydrograph.png')
fig.clf()

# plot historical range at last node, stationary synthetic range at last node and CMIP ranges at last node
baseCase = np.load('Sample1_Flows_logspace.npy')[:,-1,:]
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(range(12),np.min(np.log(CMIP5flows),0),c='#006d2c',label='CMIP5')
ax.plot(range(12),np.max(np.log(CMIP5flows),0),c='#006d2c',label='CMIP5')
ax.fill_between(range(12),np.min(np.log(CMIP5flows),0),np.max(np.log(CMIP5flows),0),color='#006d2c')

ax.plot(range(12),np.min(np.log(CMIP3flows),0),c='#2ca25f',label='CMIP3')
ax.plot(range(12),np.max(np.log(CMIP3flows),0),c='#2ca25f',label='CMIP3')
ax.fill_between(range(12),np.min(np.log(CMIP3flows),0),np.max(np.log(CMIP3flows),0),color='#2ca25f')
                
ax.plot(range(12),np.min(np.log(baseCase),0),c='#bdbdbd',label='synthetic')
ax.plot(range(12),np.max(np.log(baseCase),0),c='#bdbdbd',label='synthetic')
ax.fill_between(range(12),np.min(np.log(baseCase),0),np.max(np.log(baseCase),0),color='#bdbdbd')
             
ax.plot(range(12),np.min(np.log(np.reshape(MonthlyQ[:,-1],[105,12])),0),c='k',label='historical')
ax.plot(range(12),np.max(np.log(np.reshape(MonthlyQ[:,-1],[105,12])),0),c='k',label='historical')
ax.fill_between(range(12),np.min(np.log(np.reshape(MonthlyQ[:,-1],[105,12])),0),\
                np.max(np.log(np.reshape(MonthlyQ[:,-1],[105,12])),0),color='k')
    
ax.set_xlabel('Month',fontsize=16)
ax.set_ylabel('log(Flow at Last Node)',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
ax.set_xlim([0,11])
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
fig.legend(handles, labels)
fig.suptitle('Stationary Generator')
fig.savefig('Figs/GeneratorFigs/LogStationaryHydrograph.png')
fig.clf()

# real space, log scale
fig = plt.figure()
ax = fig.add_subplot(111)
for j in range(np.shape(baseCase)[0]):
    ax.semilogy(range(np.shape(baseCase)[1]),baseCase[j,:],c='#bdbdbd',label='Synthetic')
             
for j in range(int(np.shape(MonthlyQ)[0]/12)):
    ax.semilogy(range(12),MonthlyQ[j*12:(j+1)*12,-1],c='k',label='Historical')
    
# plot 2002 in red
ax.semilogy(range(12),MonthlyQ[93*12:94*12,-1],c='r',label='WY 2002')
    
ax.set_xlabel('Month',fontsize=16)
ax.set_ylabel('Flow at Last Node (af)',fontsize=16)
ax.set_xlim([0,11])
ax.tick_params(axis='both',labelsize=14)
ax.set_xticks(range(12))
ax.set_xticklabels(['O','N','D','J','F','M','A','M','J','J','A','S'])
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
fig.subplots_adjust(bottom=0.2)
fig.legend(handles, labels, fontsize=16,loc='lower center',ncol=3)
ax.set_title('Stationary Generator',fontsize=18)
fig.set_size_inches([7.75,5.7])
fig.savefig('Figs/GeneratorFigs/StationaryHydrograph_logscale.png')
fig.clf()

baseCaseAnnual = np.sum(baseCase,1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(np.log(baseCaseAnnual), color='b', alpha=0.5, label='Synthetic', density=True)
ax.hist(np.log(AnnualQ[:,-1]), color='g', alpha=0.5, label='Historical', density=True)
ax.set_xlabel('log(Annual Flow near State Line (af))', fontsize=16)
ax.set_ylabel('Probability Density', fontsize=16)
ax.tick_params(axis='both',labelsize=14)
fig.subplots_adjust(bottom=0.2)
fig.legend(loc='lower center',ncol=2, fontsize=16, frameon=True)
fig.set_size_inches([7.6,6.0])
plt.savefig('Figs/GeneratorFigs/StationaryHistogram.png')
plt.clf()
             
weights_base = np.ones_like(baseCaseAnnual)/float(len(baseCaseAnnual))
hist, bins = np.histogram(baseCaseAnnual, weights=weights_base)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]),len(bins))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(baseCaseAnnual, weights=weights_base, bins=logbins, color='b', alpha=0.5, label='Synthetic')

weights_hist = np.ones_like(AnnualQ[:,-1])/float(len(AnnualQ[:,-1]))
hist, bins = np.histogram(AnnualQ[:,-1], weights=weights_hist)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]),len(bins))

ax.hist(AnnualQ[:,-1], weights=weights_hist, bins=logbins, color='g', alpha=0.5, label='Historical')
ax.set_xlabel('Annual Flow near State Line (af)', fontsize=16)
ax.set_ylabel('Probability Density', fontsize=16)
ax.tick_params(axis='both',labelsize=14)
ax.set_xscale('log')
fig.subplots_adjust(bottom=0.2)
fig.legend(loc='lower center',ncol=2, fontsize=16, frameon=True)
fig.set_size_inches([7.6,6.0])
plt.savefig('Figs/GeneratorFigs/StationaryHistogram_logscale.png')
plt.clf()

# plot stationary FDCs
F_base = sortFlows(baseCase)
F_hist = sortFlows(np.reshape(MonthlyQ[:,-1],[np.shape(AnnualQ)[0],12]))
n = 12
M = np.array(range(1,n+1))
P = (M-0.5)/n

fig = plt.figure()
ax = fig.add_subplot(111)
l1, = ax.semilogy(P,np.min(F_base,0),c='#bdbdbd',label='Synthetic',zorder=1)
l2, = ax.semilogy(P,np.max(F_base,0),c='#bdbdbd',label='Synthetic',zorder=1)
ax.fill_between(P, np.min(F_base,0), np.max(F_base,0), color='#bdbdbd')
l1, = ax.semilogy(P,np.min(F_hist,0),c='k',label='Historical',zorder=1)
l2, = ax.semilogy(P,np.max(F_hist,0),c='k',label='Historical',zorder=1)
ax.fill_between(P, np.min(F_hist,0), np.max(F_hist,0), color='k')
l3, = ax.semilogy(P,F_hist[93,:],c='r',label='WY 2002',zorder=1)
    
ax.tick_params(axis='both',labelsize=14)
ax.set_xlabel('Probability of Exceedance', fontsize=16)
ax.set_ylabel('Flow at last node (af)', fontsize=16)

fig.subplots_adjust(bottom=0.2)
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.grid(True, which='both', ls='-')
fig.legend(handles, labels, fontsize=16, ncol=3, loc='lower center',frameon=True)
fig.set_size_inches([9.5,6.15])
fig.savefig('Figs/GeneratorFigs/StationaryFDC.png')
fig.clf()

# loop through scenarios and plot effects (hydrographs and annual histograms)
parameters = [r'$\mu_0$',r'$\sigma_0$',r'$\mu_1$',r'$\sigma_1$',r'$p_{00}$',r'$p_{11}$']
filenames = ['Mu_0','Sigma_0','Mu_1','Sigma_1','p_00','p_11']
for i in range(len(parameters)):
    newFlows_down = np.load('Sample' + str((i+2)*2) + '_Flows_logspace.npy')[:,-1,:]
    newFlows_up = np.load('Sample' + str((i+2)*2+1) + '_Flows_logspace.npy')[:,-1,:]
    
    uplabel = 'Increased ' + parameters[i]
    downlabel = 'Decreased ' + parameters[i]
    
    sns.set()    
    # plot hydrograph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(12), np.min(newFlows_up,0), label=uplabel, zorder=1, c='#377eb8')
    ax.plot(range(12), np.max(newFlows_up,0), label=uplabel, zorder=1, c='#377eb8')
    ax.plot(range(12), np.min(newFlows_down,0), label=downlabel, zorder=1, c='#ff7f00')
    ax.plot(range(12), np.max(newFlows_down,0), label=downlabel, zorder=1, c='#ff7f00')
    ax.plot(range(12), np.min(baseCase,0), label='Base Case', c='k')
    ax.plot(range(12), np.max(baseCase,0), label='Base Case', c='k')
                    
    ax.set_xlabel('Month',fontsize=16)
    ax.set_ylabel('Flow at last node (af)',fontsize=16)
    ax.tick_params(axis='both',labelsize=14)
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    fig.legend(handles, labels, fontsize=16, ncol=3, loc='lower center',frameon=True)
    fig.set_size_inches([9.5,6.15])
    fig.savefig('Figs/GeneratorFigs/' + filenames[i] + '_Hydrograph.png')
    fig.clf()
    
    # plot FDC
    F_new_up = sortFlows(newFlows_up)
    F_new_down = sortFlows(newFlows_down)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l1, = ax.semilogy(P, np.min(F_new_up,0), label=uplabel, zorder=1, c='#377eb8')
    l2, = ax.semilogy(P, np.max(F_new_up,0), label=uplabel, zorder=1, c='#377eb8')
    l1, = ax.semilogy(P, np.min(F_new_down,0), label=downlabel, zorder=1, c='#ff7f00')
    l2, = ax.semilogy(P, np.max(F_new_down,0), label=downlabel, zorder=1, c='#ff7f00')
    l1, = ax.semilogy(P, np.min(F_base,0), label='Base Case', zorder=1, c='k')
    l2, = ax.semilogy(P, np.max(F_base,0), label='Base Case', zorder=1, c='k')
        
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel('Probability of Exceedance', fontsize=16)
    ax.set_ylabel('Flow at last node (af)', fontsize=16)
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.grid(True, which='both', ls='-')
    fig.legend(handles, labels, fontsize=16, ncol=3, loc='lower center',frameon=True)
    fig.set_size_inches([9.5,6.15])
    fig.savefig('Figs/GeneratorFigs/' + filenames[i] + '_FDC.png')
    fig.clf()
    
    # plot histogram
    newUpAnnual = np.sum(newFlows_up,1)
    newDownAnnual = np.sum(newFlows_down,1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(np.log(newUpAnnual), label=uplabel, fill=False, histtype='step', stacked=True, \
            linewidth=2, color='#377eb8', density=True)
    ax.hist(np.log(baseCaseAnnual),density=True, label='Base Case', fill=False, histtype='step', \
            stacked=True, linewidth=2, color='k')
    ax.set_xlabel('log(Annual Flow at last node (af))',fontsize=16)
    ax.set_ylabel('Probability Density',fontsize=16)
    ax.tick_params(axis='both',labelsize=14)
    fig.subplots_adjust(bottom=0.2)
    fig.legend(loc='lower center',ncol=3,frameon=True,fontsize=16)
    fig.set_size_inches([9.5,6.15])
    fig.savefig('Figs/GeneratorFigs/' + filenames[i] + '_HistogramUp.png')
    fig.clf()
    
# plot change in probability of different dry and wet spell lengths
# with different values of transition probabilities
P = np.load('HMM_P_logspace.npy')[:,:,-1]
x = np.arange(1,15)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, ss.geom.pmf(x,1-P[0,0]), label='Base Case',c = 'k')
ax.plot(x, ss.geom.pmf(x,1-min(max(P[0,0]-0.3,0),1)), label='Decreased p_00', c='#377eb8')
ax.plot(x, ss.geom.pmf(x,1-min(max(P[0,0]+0.3,0),1)), label='Increased p_00', c='#ff7f00')
ax.set_xlabel('Length of Dry Spell (Years)',fontsize=16)
ax.set_ylabel('Probability',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
fig.subplots_adjust(bottom=0.2)
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.grid(True, which='both', ls='-')
fig.legend(handles, labels, fontsize=16, ncol=3, loc='lower center',frameon=True)
fig.set_size_inches([7.5,5.8])
fig.savefig('Figs/GeneratorFigs/DrySpellDistns.png')
fig.clf()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, ss.geom.pmf(x,1-P[0,0]), label='Base Case',c = 'k')
ax.plot(x, ss.geom.pmf(x,1-min(max(P[1,1]-0.3,0),1)), label='Decreased p_11', c='#377eb8')
ax.plot(x, ss.geom.pmf(x,1-min(max(P[1,1]+0.3,0),1)), label='Increased p_11', c='#ff7f00')
ax.set_xlabel('Length of Wet Spell (Years)',fontsize=16)
ax.set_ylabel('Probability',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
fig.subplots_adjust(bottom=0.2)
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.grid(True, which='both', ls='-')
fig.legend(handles, labels, fontsize=16, ncol=3, loc='lower center',frameon=True)
fig.set_size_inches([7.5,5.8])
fig.savefig('Figs/GeneratorFigs/WetSpellDistns.png')
fig.clf()

# log-scale
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(x, ss.geom.pmf(x,1-P[0,0]), label='Base Case',c = 'k')
ax.semilogy(x, ss.geom.pmf(x,1-min(max(P[0,0]-0.3,0),1)), label='Decreased p_00', c='#377eb8')
ax.semilogy(x, ss.geom.pmf(x,1-min(max(P[0,0]+0.3,0),1)), label='Increased p_00', c='#ff7f00')
ax.set_xlabel('Length of Dry Spell (Years)',fontsize=16)
ax.set_ylabel('Probability',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
fig.subplots_adjust(bottom=0.2)
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.grid(True, which='both', ls='-')
fig.legend(handles, labels, fontsize=16, ncol=3, loc='lower center',frameon=True)
fig.set_size_inches([7.5,5.8])
fig.savefig('Figs/GeneratorFigs/DrySpellDistns_logscale.png')
fig.clf()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(x, ss.geom.pmf(x,1-P[0,0]), label='Base Case',c = 'k')
ax.semilogy(x, ss.geom.pmf(x,1-min(max(P[1,1]-0.3,0),1)), label='Decreased p_11', c='#377eb8')
ax.semilogy(x, ss.geom.pmf(x,1-min(max(P[1,1]+0.3,0),1)), label='Increased p_11', c='#ff7f00')
ax.set_xlabel('Length of Wet Spell (Years)',fontsize=16)
ax.set_ylabel('Probability',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
fig.subplots_adjust(bottom=0.2)
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.grid(True, which='both', ls='-')
fig.legend(handles, labels, fontsize=16, ncol=3, loc='lower center',frameon=True)
fig.set_size_inches([7.5,5.8])
fig.savefig('Figs/GeneratorFigs/WetSpellDistns_logscale.png')
fig.clf()