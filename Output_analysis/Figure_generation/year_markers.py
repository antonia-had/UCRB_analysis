import scipy.stats
import numpy as np
from matplotlib import pyplot as plt


design='LHsamples_original_1000'
IDstoplot = ['3600687', '7000550', '7200799', '7200645', '5104655', '7202003']
years=[34,93] #median and 2002
colors = ["#E5E059","#EF767A"]
n=12

yr_per = lambda x, array: int(np.round(scipy.stats.percentileofscore(array, array[x], kind='mean'), decimals=0)) 

def percentiles(ID):
    HIS_short = np.loadtxt('../../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_0.txt')[:,2]
    # Reshape into water years
    # Create matrix of [no. years x no. months x no. experiments]
    f_HIS_short = np.reshape(HIS_short, (int(np.size(HIS_short)/n), n))

    # Shortage per water year
    f_HIS_short_WY = np.sum(f_HIS_short,axis=1)
    
    # Identify percentile for 2002 annual shortage
    markers=[yr_per(x,f_HIS_short_WY) for x in years]
    return markers

fig, axes = plt.subplots(2,3)
for s in range(len(axes.flat)):
    ax = axes.flat[s]
    markers = percentiles(IDstoplot[s])
    for m in range(len(markers)):
        ax.axvline(x=markers[m], linewidth=3, linestyle='--', color=colors[m])
    ax.set_xlim(0,100)
plt.show()
