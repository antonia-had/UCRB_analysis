from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
plt.switch_backend('agg')
plt.ioff()

design = str(sys.argv[1])

## load paleo data at Cisco
#Paleo = pd.read_csv('../Summary_info/Cisco_Recon_v_Observed_v_Stateline.csv')
#
## re-scale Cisco data to estimate data at CO-UT state line
#factor = np.nanmean(Paleo['ObservedNaturalStateline']/Paleo['ObservedNaturalCisco'])
#Paleo['ScaledNaturalCisco'] = Paleo['ObservedNaturalCisco']*factor
#Paleo['ScaledReconCisco'] = Paleo['ReconCisco']*factor

years = np.arange(1909, 2014)
years_s = np.arange(1950, 2014)

# Load historic data
#historic_file = open('../Summary_info/cm2015x.xbm', 'r')
#all_split_data = [x.split('.') for x in historic_file.readlines()]
#historic_data = np.zeros([len(years),12])
#count = 0
#for i in range(16, len(all_split_data)):
#    row_data = []
#    row_data.extend(all_split_data[i][0].split())
#    if row_data[1] == '09163500':
#        data_to_write = [row_data[2]]+all_split_data[i][1:12]
#        historic_data[count,:] = [int(j) for j in data_to_write]
#        count+=1
#        
#np.save('../Summary_info/historic_flows.npy', historic_data)
historic_data = np.load('../Summary_info/historic_flows.npy')
        
# Load CMIP flows
CMIP3_flows = np.genfromtxt('../Summary_info/CMIP3_flows.csv', delimiter=',')
CMIP3_flows = np.reshape(CMIP3_flows, [112, 64,12])
CMIP5_flows = np.genfromtxt('../Summary_info/CMIP5_flows.csv', delimiter=',')
CMIP5_flows = np.reshape(CMIP5_flows, [97, 64,12])

# Load synthetic stationary flows
baseCase = np.load('../Summary_info/Sample1_Flows_logspace.npy')[:,-1,:]

# Load synthetic nonstationary runs
#synthetic_flows = np.zeros([10000, len(years),12])
#for s in range(1000):
#    for k in range(10):       
#        synthetic_file = open('../' + design + '/Experiment_files/cm2015x_S'+str(s+1)+'_'+str(k+1)+'.xbm', 'r')
#        all_split_data = [x.split('.') for x in synthetic_file.readlines()]
#        yearcount = 0
#        for i in range(16, len(all_split_data)):
#            row_data = []
#            row_data.extend(all_split_data[i][0].split())
#            if row_data[1] == '09163500':
#                data_to_write = [row_data[2]]+all_split_data[i][1:12]
#                synthetic_flows[s*10+k,yearcount,:] = [int(j) for j in data_to_write]
#                yearcount+=1
            
#np.save('../Summary_info/'+design+'_flows.npy', synthetic_flows)

synthetic_flows = np.load('../Summary_info/'+design+'_flows.npy')

colors = ['#DD7373', '#305252', '#3C787E','#D0CD94', '#9597a3'] #'#AA1209'
labels=['Historic', 'Stationary synthetic', 'CMIP3', 'CMIP5', 'This experiment'] #'Paleo'
data = [np.sum(historic_data, axis=1), np.sum(baseCase, axis=1), np.sum(CMIP3_flows, axis=2), np.sum(CMIP5_flows, axis=2), np.sum(synthetic_flows, axis=2)] #Paleo['ScaledReconCisco'][:429].values

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
violinplots=ax.violinplot(data, vert=True)
violinplots['cbars'].set_edgecolor('black')
violinplots['cmins'].set_edgecolor('black')
violinplots['cmaxes'].set_edgecolor('black')
for i in range(len(violinplots['bodies'])):
    vp = violinplots['bodies'][i]
    vp.set_facecolor(colors[i])
    vp.set_edgecolor('black')
    vp.set_alpha(1)
ax.set_ylabel('Flow at Last Node (af)',fontsize=20)
ax.set_xticks(np.arange(1,6))
ax.set_xticklabels(labels,fontsize=16)
plt.savefig('../Summary_info/streamflow_violinplot_'+design+'.svg')

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
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
ax.set_ylabel('Flow at Last Node (af)',fontsize=20)
ax.set_xticks(np.arange(1,6))
ax.set_xticklabels(labels,fontsize=16)
plt.savefig('../Summary_info/streamflow_violinplot_'+design+'_log.svg')

