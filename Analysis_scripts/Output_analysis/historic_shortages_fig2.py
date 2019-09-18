import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
import matplotlib.font_manager as font_manager

months = 12
years = 105
demands = pd.read_csv('../../Structures_files/demands.csv',index_col=0)
shortages = pd.read_csv('../../Structures_files/shortages.csv',index_col=0)
structures = demands.index.tolist()
    
yearly_demands = np.reshape(demands.values, (len(structures), years, months)) 
yearly_shortages = np.reshape(shortages.values, (len(structures), years, months)) 

yearly_demands_sorted = np.sort(yearly_demands, axis=2)
yearly_shortages_sorted = np.sort(yearly_shortages, axis=2)

M = np.array(range(1,months+1))
P = (M-0.5)/months
p=np.arange(0,100,100/months)

IDstoplot = ['3600687', '7000550', '7200799', '7200645', '3704614', '7202003']

font = font_manager.FontProperties(family='Gill Sans MT',
                                   style='normal', size=16)

fig, axes = plt.subplots(2,3)
for s in range(len(axes.flat)):
    ax = axes.flat[s]
    j = structures.index(IDstoplot[s])
    for i in range(years):
        ax.plot(p, yearly_shortages_sorted[j,i,:], color='black')
    ax.plot(p, yearly_shortages_sorted[j,93,:], color='red', linewidth=4, label='Water Year 2002')
    if s>2:
        ax.set_xlabel('Shortage magnitude percentile',fontsize=20, fontname = 'Gill Sans MT')
    if s==0 or s==3:
        ax.set_ylabel('Monthly shortage (af)',fontsize=20, fontname = 'Gill Sans MT')
    ax.set_yscale('symlog')
    ax.set_ylim(-0.01, 40500)
    ax.tick_params(axis='both',labelsize=14)
    if s==0:
        ax.legend(loc = 'upper left', prop = font)
fig.set_size_inches([20,10])
fig.savefig('./Paper1_figures/historic_impacts_fig2.svg')
fig.savefig('./Paper1_figures/historic_impacts_fig2.png')
