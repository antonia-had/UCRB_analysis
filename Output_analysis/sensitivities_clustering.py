from sklearn import cluster
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

indices = pd.read_csv('./Figure_generation/2002_S1_freq_clustering.csv',index_col=0)

rights = pd.read_csv('../Structures_files/diversions_admin.csv')
groupbyadmin = rights.groupby('WDID')
right_groups_admin = groupbyadmin.apply(lambda x: x['Admin'].unique())
right_groups_decree = groupbyadmin.apply(lambda x: list(x['Decree']))
agg_rights = pd.DataFrame(np.zeros([len(right_groups_admin),2]), 
                          index = right_groups_admin.index,
                          columns=['Admin', 'Decree'])
for i in range(len(right_groups_admin)):
    agg_rights.at[right_groups_admin.index[i]] = [np.mean(right_groups_admin[i]), np.sum(right_groups_decree[i])]
agg_rights = agg_rights.reindex(list(indices.index))
#agg_rights['WD'] = [int(ID[:2]) for ID in list(indices.index)]


#'''Cluster on sensitivities
#'''
#values = indices.values
#distortions = []
#for i in range(1, 15):
#    km = cluster.KMeans(n_clusters=i, init='k-means++', n_init=50, max_iter=1000, algorithm='full')
#    km.fit(values)
#    distortions.append(km.inertia_)
## plot
#plt.plot(range(1, 15), distortions, marker='o')
#plt.xlabel('Number of clusters')
#plt.ylabel('Distortion')
#plt.show()
#
#ngroups = 4
#k_means = cluster.KMeans(n_clusters=ngroups, init='k-means++', n_init=50, max_iter=1000, algorithm='full')
#y_km = k_means.fit_predict(values)
#
#'''Pairwise plots
#'''
#pd.plotting.scatter_matrix(indices, c=y_km, figsize=(15, 15), marker='o',hist_kwds={'bins': 10}, s=50, alpha=.8)
#pd.plotting.scatter_matrix(agg_rights, c=y_km, figsize=(15, 15), marker='o',hist_kwds={'bins': 10}, s=50, alpha=.8)


'''Cluster on water rights. 
Need to normalize first.
'''
values = agg_rights.values
mins = values.min(axis=0).clip(0)
maxs = values.max(axis=0).clip(0,1)
normalized = values.copy()
for i in range(len(values[0,:])):
    mm = values[:,i].min()
    mx = values[:,i].max()
    if mm!=mx:
        normalized[:,i] = (values[:,i] - mm) / (mx - mm)
    else:
        normalized[:,i] = 0
#distortions = []
#for i in range(1, 15):
#    km = cluster.KMeans(n_clusters=i, init='k-means++', n_init=50, max_iter=1000, algorithm='full')
#    km.fit(normalized)
#    distortions.append(km.inertia_)
## plot
#plt.plot(range(1, 15), distortions, marker='o')
#plt.xlabel('Number of clusters')
#plt.ylabel('Distortion')
#plt.show()

ngroups = 4
k_means = cluster.KMeans(n_clusters=ngroups, init='k-means++', n_init=50, max_iter=1000, algorithm='full')
y_km = k_means.fit_predict(normalized)

'''Pairwise plots
'''
axes = pd.plotting.scatter_matrix(indices, c='royalblue', figsize=(15, 15), marker='o',hist_kwds={'bins': 10}, s=50, alpha=.8)
pd.plotting.scatter_matrix(agg_rights, c=y_km, figsize=(15, 15), marker='o',hist_kwds={'bins': 10}, s=50, alpha=.8)
pd.plotting.scatter_matrix(indices, c=y_km, figsize=(15, 15), marker='o',hist_kwds={'bins': 10}, s=50, alpha=.8)
for i in range(7):
    for j in range(7):
        if i != j:
             axes[i,j].set_ylim(0.0,1.0) 
