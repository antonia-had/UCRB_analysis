import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

MonthlyQ_s = np.load('../Sample1_Flows_logspace.npy') # nyears x nsites x 12
MonthlyQ_s = np.reshape(np.swapaxes(MonthlyQ_s,1,2),[105*12,208]) # nyears*12 x nsites

MonthlyQ_h = np.loadtxt('../MonthlyQ.csv',skiprows=1,usecols=np.arange(1,209),delimiter=',') # nyears*12 x nsites

cmap = matplotlib.cm.get_cmap('viridis')

sns.set_style("darkgrid")

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.matshow(np.corrcoef(np.transpose(MonthlyQ_h)),cmap=cmap)
sm = matplotlib.cm.ScalarMappable(cmap=cmap)
sm.set_array([np.min(np.corrcoef(np.transpose(MonthlyQ_h))),np.max(np.corrcoef(np.transpose(MonthlyQ_h)))])
ax.set_title('Historical Spatial Correlation',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
ax.set_ylabel('Basin Node',fontsize=16)

ax = fig.add_subplot(1,2,2)
ax.matshow(np.corrcoef(np.transpose(MonthlyQ_s)),cmap=cmap)
sm = matplotlib.cm.ScalarMappable(cmap=cmap)
sm.set_array([np.min(np.corrcoef(np.transpose(MonthlyQ_s))),np.max(np.corrcoef(np.transpose(MonthlyQ_s)))])
ax.set_title('Synthetic Spatial Correlation',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
ax.set_ylabel('Basin Node',fontsize=16)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
fig.axes[-1].set_ylabel('Pearson Correlation Coefficient',fontsize=16)
cbar_ax.tick_params(labelsize=14)
fig.set_size_inches(14,6.5)
fig.savefig('FigureS4.pdf')
fig.clf()
