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

basinshortages = np.sum(yearly_shortages,axis=0)
medianyear = np.argsort(basinshortages)[len(basinshortages)//2]

rights = pd.read_csv('../../Structures_files/diversions_admin.csv')
groupbyadmin = rights.groupby('WDID')
right_groups_admin = groupbyadmin.apply(lambda x: x['Admin'].unique())
right_groups_decree = groupbyadmin.apply(lambda x: list(x['Decree']))
right_groups_source = groupbyadmin.apply(lambda x: x['Source'].unique())
agg_rights = pd.DataFrame(np.zeros([len(right_groups_admin),3]), 
                          index = right_groups_admin.index,
                          columns=['Admin', 'Decree','Source'])
for i in range(len(right_groups_admin)):
    agg_rights.at[right_groups_admin.index[i]] = [np.mean(right_groups_admin[i]), np.sum(right_groups_decree[i]), right_groups_source[i][0]]
agg_rights = agg_rights.reindex(list(demands.index))
agg_rights['WD'] = [int(ID[:2]) for ID in list(agg_rights.index)]
wdcounts = agg_rights['WD'].value_counts().sort_index()

indices = pd.read_csv('median_DELTA_max_duration.csv',index_col=0)
indices.insert(0, 'N/A', 0)
indices['1stFactor'] = indices.idxmax(axis=1)
indices = indices.reindex(list(demands.index))
colors = [color_list[param_names.index(f)] for f in indices['1stFactor'].values]

ratio2002 = np.divide(yearly_shortages[:,medianyear],yearly_demands[:,medianyear],
                      out=np.zeros_like(yearly_shortages[:,medianyear]), where=yearly_demands[:,medianyear]!=0)
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
plt.show()
