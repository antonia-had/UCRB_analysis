import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
plt.ioff()

months = 12
years = 105
demands = pd.read_csv('../../Structures_files/demands.csv',index_col=0)
shortages = pd.read_csv('../../Structures_files/shortages.csv',index_col=0)
structures = demands.index.tolist()

rights = pd.read_csv('./Model_comparison/diversions_admin.csv')
groups = rights.groupby('WDID')
right_groups = groups.apply(lambda x: x['Admin'].unique())
agg_rights = pd.DataFrame(np.zeros([len(right_groups),2]), index = right_groups.index)
for i in range(len(right_groups)):
    agg_rights.at[right_groups.index[i],0] = np.mean(right_groups[i])
right_max=100000#agg_rights[0].max()
right_min=9324#agg_rights[0].min()
for i in range(len(right_groups)):
    agg_rights.at[right_groups.index[i],1] = (agg_rights.at[right_groups.index[i],0] - right_min) / (right_max - right_min)
    
yearly_demands = np.reshape(demands.values, (len(structures), years, months)) 
yearly_shortages = np.reshape(shortages.values, (len(structures), years, months)) 
yearly_ratios = np.nan_to_num(yearly_shortages / yearly_demands)

yearly_demands_sorted = np.sort(yearly_demands, axis=2)
yearly_shortages_sorted = np.sort(yearly_shortages, axis=2)

M = np.array(range(1,months+1))
P = (M-0.5)/months
p=np.arange(0,100,100/months)

#for s in range(len(structures)):
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    for i in range(years):
#        ax.plot(p, yearly_shortages_sorted[s,i,:], color='black')
#    ax.plot(p, yearly_shortages_sorted[s,93,:], color='red',label='Water Year 2002')  
#    ax.set_xlabel('Shortage magnitude percentile',fontsize=16)
#    ax.set_ylabel('Monthly shortage (af)',fontsize=16)
#    ax.tick_params(axis='both',labelsize=14)
#    ax.set_title('Monthly historical shortages for user '+ structures[s],fontsize=18)
#    ax.legend()
#    fig.set_size_inches([12,9])
#    fig.savefig('Historic_impacts_all_users/'+ structures[s]+'.svg')
#    fig.savefig('Historic_impacts_all_users/'+ structures[s]+'.png')

def shortage_duration(sequence):
    cnt_shrt = [sequence[i]>0.2 for i in range(len(sequence))] # Returns a list of True values when there's a shortage
    shrt_dur = [ sum( 1 for _ in group ) for key, group in itertools.groupby( cnt_shrt ) if key ] # Counts groups of True values
    return shrt_dur


metrics_labels = ['Mean shortage ratio', 
               'Worst case shortage', 
               'Number of shortages above 20%', 
               'Worst multi-year shortage'] 

metrics = np.zeros([len(structures), len(metrics_labels)])

for i in range(len(structures)):
    metrics[i,0]=np.mean(yearly_ratios[i,:,:])
    metrics[i,1]=np.max(yearly_ratios[i,:,:])
    metrics[i,2]=np.sum(np.greater_equal(yearly_ratios[i,:,:], 0.3))
    sustained_duration = shortage_duration(np.mean(yearly_ratios[i,:,:], axis=1))
    if sustained_duration:
        metrics[i,3]=np.max(sustained_duration)
    else: 
        metrics[i,3]=0
# Normalization across objectives
mins = metrics.min(axis=0)
maxs = metrics.max(axis=0)
norm_metrics = metrics.copy()
for i in range(4):
    mm = metrics[:,i].min()
    mx = metrics[:,i].max()
    if mm!=mx:
        norm_metrics[:,i] = (metrics[:,i] - mm) / (mx - mm)
    else:
        norm_metrics[:,i] = 1

cmap = plt.cm.plasma


fig = plt.figure(figsize=(18,9)) # create the figure
ax = fig.add_subplot(1, 1, 1)    # make axes to plot on

## Plot all solutions
for i in range(len(structures)):
    ys = norm_metrics[i,:]
    xs = range(len(ys))
    ax.plot(xs, ys, c=cmap(agg_rights.at[right_groups.index[i],1]), linewidth=2)

#Colorbar
sm = matplotlib.cm.ScalarMappable(cmap=cmap)
sm.set_array([metrics[:,0].min(),metrics[:,0].max()])
cbar = fig.colorbar(sm)
cbar.ax.set_ylabel("\nRight seniority (0 most senior)")

# Tick values
minvalues = ["{0:.3f}".format(mins[0]), "{0:.3f}".format(-mins[1]),str(-mins[2]), str(mins[3]), str(0)]
maxvalues = ["{0:.2f}".format(maxs[0]), "{0:.3f}".format(-maxs[1]),str(-maxs[2]), "{0:.2f}".format(maxs[3]), str(0) ]

ax.set_ylabel("<- Preference", size= 12)
ax.set_yticks([])
ax.set_xticks([0,1,2,3])
ax.set_xticklabels([minvalues[i]+'\n'+metrics_labels[i] for i in range(len(metrics_labels))])
#make a twin axis for toplabels
ax1 = ax.twiny()
ax1.set_yticks([])
ax1.set_xticks([0,1,2,3])
ax1.set_xticklabels([maxvalues[i] for i in range(len(maxs))])
plt.show()
