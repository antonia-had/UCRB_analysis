from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
plt.switch_backend('agg')
plt.ioff()


years = np.arange(1909, 2014)
years_s = np.arange(1950, 2014)

# Load historic data
historic_file = open('./Experiment_files/cm2015x.xbm', 'r')
all_split_data = [x.split('.') for x in historic_file.readlines()]
historic_data = np.zeros([len(years),12])
count = 0
for i in range(16, len(all_split_data)):
    row_data = []
    row_data.extend(all_split_data[i][0].split())
    if row_data[1] == '09163500':
        data_to_write = [row_data[2]]+all_split_data[i][1:12]
        historic_data[count,:] = [int(j) for j in data_to_write]
        count+=1

CMIP3_flows = np.genfromtxt('CMIP3_flows.csv', delimiter=',')
CMIP3_flows = np.reshape(CMIP3_flows, [112, 64,12])
CMIP5_flows = np.genfromtxt('CMIP5_flows.csv', delimiter=',')
CMIP5_flows = np.reshape(CMIP5_flows, [97, 64,12])


# Load synthetic runs
synthetic_flows = np.zeros([10000, len(years),12])
for s in range(1000):
    for k in range(10):       
        synthetic_file = open('./Experiment_files/cm2015x_S'+str(s+1)+'_'+str(k+1)+'.xbm', 'r')
        all_split_data = [x.split('.') for x in synthetic_file.readlines()]
        yearcount = 0
        for i in range(16, len(all_split_data)):
            row_data = []
            row_data.extend(all_split_data[i][0].split())
            if row_data[1] == '09163500':
                data_to_write = [row_data[2]]+all_split_data[i][1:12]
                synthetic_flows[s*10+k,yearcount,:] = [int(j) for j in data_to_write]
                yearcount+=1
            
synthetic_flows = np.delete(synthetic_flows,75,1)
        
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.fill_between(range(12), np.min(np.min(synthetic_flows, axis=0),axis=0),
#                np.max(np.max(synthetic_flows, axis=0),axis=0), color='#6B6D76',label='This experiment', alpha = 0.5)
ax.fill_between(range(12), np.min(np.min(CMIP3_flows, axis=0),axis=0),
                np.max(np.max(CMIP3_flows, axis=0),axis=0), color='#D0CD94',label='CMIP3', alpha = 0.5)
ax.fill_between(range(12), np.min(np.min(CMIP5_flows, axis=0),axis=0),
                np.max(np.max(CMIP5_flows, axis=0),axis=0), color='#3C787E',label='CMIP5', alpha = 0.5)
#ax.fill_between(range(12), np.min(historic_data, axis=0),
#                np.max(historic_data, axis=0), color='#03191E',label='Historical')
#ax.plot(range(12), historic_data[93,:], color='#AA1209',label='Water Year 2002')             
ax.set_yscale( "log" )               
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
ax.set_title('Streamflow across experiments',fontsize=18)
fig.set_size_inches([12,9])
fig.savefig('Experiment_comparison.svg')
plt.close()

