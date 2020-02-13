import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


months = 12
years = 105
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'ShoshoneDMND','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift']
color_list = ["#F18670", "#E24D3F", "#CF233E", "#681E33", "#676572", "#F3BE22", "#59DEBA", "#14015C", "#DAF8A3", "#0B7A0A", "#F8FFA2", "#578DC0", "#4E4AD8", "#32B3F7","#F77632"]  
area_colors = ["#F4FAFF","white"]
        

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
wdcounts = agg_rights['WD'].value_counts().sort_index()

indices = pd.read_csv('2002_DELTA_freq.csv',index_col=0)
indices['1stFactor'] = indices.idxmax(axis=1)
colors = [color_list[param_names.index(f)] for f in indices['1stFactor'].values]

ratio2002 = np.divide(yearly_shortages[:,93],yearly_demands[:,93],
                      out=np.zeros_like(yearly_shortages[:,93]), where=yearly_demands[:,93]!=0)
#agg_rights['Source'] = agg_rights['Source'].fillna('')

#order=np.argsort(agg_rights['WD'].values)
order=np.argsort(agg_rights['Admin'].values)
#order=np.argsort(ratio2002)

ratio2002_sorted = ratio2002[order]
colors_sorted = [colors[i] for i in order]

iN = len(ratio2002_sorted)
width = (2*np.pi)/iN
theta=np.arange(0,2*np.pi-width,width)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0.1, 0.1, 0.75, 0.75], polar=True)
bars = ax.bar(theta, ratio2002_sorted, color=colors_sorted, width=width,zorder=5)
#total=0
#for i in range(len(wdcounts.values)):
#    count=wdcounts.values[i]
#    total+=count
#    ax.fill_between(np.linspace(2*np.pi*(total-count)/iN,2*np.pi*(total/iN),100),0,1, facecolor=area_colors[np.mod(i,2)],zorder=1)
#    ax.axvline(2*np.pi*(total/iN),linewidth=1,linestyle = '--', color='gray',zorder=2)
ax.set_xticks([0])
ax.set_ylim(0,1)
#ax.get_xaxis().set_visible(False)
