import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
import matplotlib.font_manager as font_manager

months = 12
years = 105
all_IDs = np.genfromtxt('../../Structures_files/metrics_structures.txt',dtype='str').tolist()
demands = pd.read_csv('../../Summary_info/demands_uncurtailed.csv',header = None, index_col=False)
shortages = pd.read_csv('../../Summary_info/shortages_uncurtailed.csv',header = None, index_col=False)
demands['index']=all_IDs
demands = demands.set_index('index')
shortages['index']=all_IDs
shortages = shortages.set_index('index')
    
yearly_demands = np.reshape(demands.values, (len(all_IDs), years, months)) 
yearly_shortages = np.reshape(shortages.values, (len(all_IDs), years, months)) 

yearly_demands_sorted = np.sort(yearly_demands, axis=2)
yearly_shortages_sorted = np.sort(yearly_shortages, axis=2)
yearly_shortages_sorted_SI = yearly_shortages_sorted*1233.4818

M = np.array(range(1,months+1))
P = (M-0.5)/months
p=np.arange(0,100,100/months)

IDstoplot = ['3600687', '7000550', '7200799', '7200645', '3704614', '7202003']

font = font_manager.FontProperties(family='Gill Sans MT',
                                   style='normal', size=16)
#Figure in imperial units
fig, axes = plt.subplots(2,3)
for s in range(len(axes.flat)):
    ax = axes.flat[s]
    j = all_IDs.index(IDstoplot[s])
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

#Figure in metric units
fig, axes = plt.subplots(2,3)
for s in range(len(axes.flat)):
    ax = axes.flat[s]
    j = all_IDs.index(IDstoplot[s])
    for i in range(years):
        ax.plot(p, yearly_shortages_sorted_SI[j,i,:], color='black')
    ax.plot(p, yearly_shortages_sorted_SI[j,93,:], color='red', linewidth=4, label='Water Year 2002')
    if s>2:
        ax.set_xlabel('Shortage magnitude percentile',fontsize=20, fontname = 'Gill Sans MT')
    if s==0 or s==3:
        ax.set_ylabel('Monthly shortage ($m^3$)',fontsize=20, fontname = 'Gill Sans MT')
    ax.set_yscale('symlog')
    ax.set_ylim(-0.01, 40500)
    ax.tick_params(axis='both',labelsize=14)
    if s==0:
        ax.legend(loc = 'upper left', prop = font)
fig.set_size_inches([20,10])
fig.savefig('./Paper1_figures/historic_impacts_fig2_SI.svg')
fig.savefig('./Paper1_figures/historic_impacts_fig2_SI.png')

