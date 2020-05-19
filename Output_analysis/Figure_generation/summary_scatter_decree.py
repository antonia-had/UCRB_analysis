import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


months = 12
years = 105
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'ShoshoneDMND','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift', 'Interactions', 'N/A']
color_list = ["#ff8000", "#b15a29", "#17BECF", "#ffff98", 
              "#7B4173", "#31A354", "#fcbd6d", "#e2171a", "#f99998", 
              "#1F77B4", "#AEC7E8", "#843C39", "#104162", "#BD9E39","#D9D9D9","black"] 
area_colors = ['white','lightgrey']          

all_IDs = np.genfromtxt('../../Structures_files/metrics_structures_old.txt',dtype='str').tolist()
demands = pd.read_csv('../../Summary_info/demands_uncurtailed.csv',header = None, index_col=False)
shortages = pd.read_csv('../../Summary_info/shortages_uncurtailed.csv',header = None, index_col=False)
demands['index']=all_IDs
demands = demands.set_index('index')
shortages['index']=all_IDs
shortages = shortages.set_index('index')

demands = demands.loc[~demands.index.duplicated(keep='first')]
shortages = shortages.loc[~shortages.index.duplicated(keep='first')]

yearly_demands = np.sum(np.reshape(demands.values, (len(demands), years, months)), axis=2)
yearly_shortages = np.sum(np.reshape(shortages.values, (len(demands), years, months)), axis=2)

rights = pd.read_csv('../../Structures_files/diversions_admin.csv')
groupbyadmin = rights.groupby('WDID')
right_groups_admin = groupbyadmin.apply(lambda x: list(x['Admin']))
right_groups_decree = groupbyadmin.apply(lambda x: list(x['Decree']))
right_groups_source = groupbyadmin.apply(lambda x: x['Source'].unique())
agg_rights = pd.DataFrame(np.zeros([len(right_groups_admin),3]), 
                          index = right_groups_admin.index,
                          columns=['Admin', 'Decree','Source'])
for i in range(len(right_groups_admin)):
    weights=right_groups_decree[i] 
    if np.sum(weights)>0:
        entries = [np.average(right_groups_admin[i],weights=weights)]
    else:
        entries = [99999.99999]
    entries.extend((np.sum(right_groups_decree[i]), right_groups_source[i][0]))
    agg_rights.at[right_groups_admin.index[i]] = entries
agg_rights = agg_rights.reindex(list(demands.index))
agg_rights['WD'] = [int(ID[:2]) for ID in list(agg_rights.index)]

indices = pd.read_csv('2002_DELTA_max_duration.csv',index_col=0)
indices.insert(0, 'N/A', 0)
indices['1stFactor'] = indices.idxmax(axis=1)
indices = indices.reindex(list(demands.index))
colors = [color_list[param_names.index(f)] for f in indices['1stFactor'].values]

ratio2002 = np.divide(yearly_shortages[:,93],yearly_demands[:,93],
                      out=np.zeros_like(yearly_shortages[:,93]), where=yearly_demands[:,93]!=0)

priority_order=np.argsort(agg_rights['Admin'].values)
decrees = agg_rights['Decree'].values[priority_order]
colors_ordered = [colors[i] for i in priority_order]
ratio_ordered = ratio2002[priority_order]

# fig = plt.figure(figsize=(8, 8))
# ax = plt.axes()
# ax.scatter(np.arange(len(priority_order)), decrees, c=colors_ordered, s=ratio_ordered*500,zorder=5)
# # ax.set_yticks(ticks)
# # ax.set_yticklabels(wdcounts.index.values)
# ax.tick_params(axis='both',labelsize='16')
# ax.set_xlabel('Water right priority rank',size=20)
# ax.set_ylabel('Diversion decree',size=20)
# ax.set_xlim(0,len(priority_order))
# # ax.set_ylim(0,len(priority_order))
# plt.yscale('symlog')

fig,(ax1,ax2,ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2, 3]})
ax1.set_ylim(1900,1950)
ax2.set_ylim(975,1080)
ax3.set_ylim(-50,650)

upper_yticks = np.arange(1900,2000,50) 
ax1.set_yticks(upper_yticks)

middle_yticks = np.arange(975,1080,50) 
ax2.set_yticks(middle_yticks)

lower_yticks = np.arange(0,650,100)
ax3.set_yticks(lower_yticks)

ax1.scatter(np.arange(len(priority_order)), decrees, c=colors_ordered, s=ratio_ordered*500,zorder=5)
ax2.scatter(np.arange(len(priority_order)), decrees, c=colors_ordered, s=ratio_ordered*500,zorder=5)
ax3.scatter(np.arange(len(priority_order)), decrees, c=colors_ordered, s=ratio_ordered*500,zorder=5)

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)

# ax1.xaxis.tick_top() 
# ax1.xaxis.tick_bottom()
# ax2.xaxis.tick_top() 
# ax3.xaxis.tick_bottom()
plt.subplots_adjust(hspace=0.2) 

ax1.tick_params(axis='both',labelsize='16')
ax2.tick_params(axis='both',labelsize='16')
ax3.tick_params(axis='both',labelsize='16')

ax3.set_xlabel('Water right priority rank',size=20)
ax2.set_ylabel('Diversion decree',size=20)

ax1.set_xlim(0,len(priority_order))
ax2.set_xlim(0,len(priority_order))
ax3.set_xlim(0,len(priority_order))



