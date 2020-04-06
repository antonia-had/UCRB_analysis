import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
import pandas as pd



indices = pd.read_csv('2002_DELTA_freq.csv',index_col=0)
irrigation = np.genfromtxt('../../Structures_files/irrigation.txt',dtype='str').tolist()
values = indices.values
params = list(indices.columns)
IDs = list(indices.index)

# Create a mask with brushing conditions
#mask = [True if reference[i,1]>=-0.2 else False for i in range(len(reference[:,1]))]
#reference = reference[mask,:] 

 # Constraint (always 0)

# Normalization across objectives
mins = values.min(axis=0).clip(0)
maxs = values.max(axis=0).clip(0,1)
norm_reference = values.copy()
for i in range(len(params)):
    mm = values[:,i].min()
    mx = values[:,i].max()
    if mm!=mx:
        norm_reference[:,i] = (values[:,i] - mm) / (mx - mm)
    else:
        norm_reference[:,i] = 0
        
indicesnorm = pd.DataFrame(data=norm_reference, columns=indices.columns, index=indices.index)

cmap = matplotlib.cm.get_cmap("Blues")

fig = plt.figure(figsize=(18,9)) # create the figure
ax = fig.add_subplot(1, 1, 1)    # make axes to plot on

## Plot all solutions
for ID in IDs:
    ys = indicesnorm.loc[ID].values
    xs = range(len(ys))
    ax.plot(xs, ys, c='royalblue', linewidth=2)
    
## Plot all solutions
for ID in irrigation:
    ys = indicesnorm.loc[ID].values
    xs = range(len(ys))
    ax.plot(xs, ys, c='orange', linewidth=2)

# Tick values
minvalues = ["{0:.3f}".format(mins[0]), "{0:.3f}".format(-mins[1]), 
             str(-mins[2]), "{0:.3f}".format(-mins[3]), "{0:.2f}".format(-mins[4]), str(0)]
maxvalues = ["{0:.2f}".format(maxs[0]), "{0:.3f}".format(-maxs[1]), 
             str(-maxs[2]), "{0:.2f}".format(maxs[3]), "{0:.2f}".format(-maxs[4]), str(0) ]

ax.set_ylabel("Delta method index value", size= 12)
ax.set_yticks([])
ax.set_xticks(list(np.arange(len(params))))
ax.set_xticklabels(params)
#make a twin axis for toplabels
ax1 = ax.twiny()
ax1.set_yticks([])
ax1.set_xticks(list(np.arange(len(params))))
ax1.set_xticklabels(["{0:.2f}".format(maxs[i]) for i in range(len(maxs))])

##Colorbar
#sm = matplotlib.cm.ScalarMappable(cmap=cmap)
#sm.set_array([reference[:,0].min(),reference[:,0].max()])
#cbar = fig.colorbar(sm)
#cbar.ax.set_ylabel("\nNet present value (NPV)")

#plt.savefig('Objectives_parallel_axis_brushed.svg')
#plt.savefig('Objectives_parallel_axis_brushed.png')