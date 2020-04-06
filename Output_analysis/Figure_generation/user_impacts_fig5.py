import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches
from scipy import stats
import itertools
import sys
import matplotlib.font_manager as font_manager
plt.ioff()

font = font_manager.FontProperties(family='Gill Sans MT',
                                   style='normal', size=16)

design = str(sys.argv[1])

all_IDs = ['3600687', '7000550', '7200799', '7200645', '3704614', '7202003']
percentiles = np.arange(0,100)
samples = 1000
realizations = 10
idx = np.arange(2,22,2)

def alpha(i, base=0.2):
    l = lambda x: x+base-x*base
    ar = [l(0)]
    for j in range(i):
        ar.append(l(ar[-1]))
    return ar[-1]

def shortage_duration(sequence):
    cnt_shrt = [sequence[i]>0 for i in range(len(sequence))] # Returns a list of True values when there's a shortage
    shrt_dur = [ sum( 1 for _ in group ) for key, group in itertools.groupby( cnt_shrt ) if key ] # Counts groups of True values
    return shrt_dur
  
def plotSDC(ax1, synthetic, histData):
    n = 12
    #Reshape historic data to a [no. years x no. months] matrix
    f_hist = np.reshape(histData, (int(np.size(histData)/n), n))
    #Reshape to annual totals
    f_hist_totals = np.sum(f_hist,1)  
    #Calculate historical shortage duration curves
    F_hist = np.sort(f_hist_totals) # for inverse sorting add this at the end [::-1]
    
    #Reshape synthetic data
    #Create matrix of [no. years x no. months x no. samples]
    synthetic_global = np.zeros([int(np.size(histData)/n),n,samples*realizations]) 
    # Loop through every SOW and reshape to [no. years x no. months]
    for j in range(samples*realizations):
        synthetic_global[:,:,j]= np.reshape(synthetic[:,j], (int(np.size(synthetic[:,j])/n), n))
    #Reshape to annual totals
    synthetic_global_totals = np.sum(synthetic_global,1)
    print(np.shape(synthetic_global_totals))
    
    p=np.arange(100,-10,-10)

    #Calculate synthetic shortage duration curves
    F_syn = np.empty([int(np.size(histData)/n),samples*realizations])
    F_syn[:] = np.NaN
    for j in range(samples*realizations):
        F_syn[:,j] = np.sort(synthetic_global_totals[:,j])
    
    # For each percentile of magnitude, calculate the percentile among the experiments ran
    perc_scores = np.zeros_like(F_syn) 
    for m in range(int(np.size(histData)/n)):
        perc_scores[m,:] = [stats.percentileofscore(F_syn[m,:], j, 'rank') for j in F_syn[m,:]]
                
    P = np.arange(1.,len(F_hist)+1)*100 / len(F_hist)
    
    ylimit = round(np.max(F_syn))
    # ax1
    handles = []
    labels=[]
    color = '#000292'
    for i in range(len(p)):
        ax1.fill_between(P, np.min(F_syn[:,:],1), np.percentile(F_syn[:,:], p[i], axis=1), color=color, alpha = 0.1)
        ax1.plot(P, np.percentile(F_syn[:,:], p[i], axis=1), linewidth=0.5, color=color, alpha = 0.3)
        handle = matplotlib.patches.Rectangle((0,0),1,1, color=color, alpha=alpha(i, base=0.1))
        handles.append(handle)
        label = "{:.0f} %".format(100-p[i])
        labels.append(label)
    ax1.plot(P,F_hist, c='black', linewidth=2, label='Historical record')
    ax1.set_ylim(0,ylimit)
    ax1.set_xlim(0,100)
    ax1.legend(handles=handles, labels=labels, framealpha=1, fontsize=8, loc='upper left', title='Frequency in experiment',ncol=2)
    ax1.set_xlabel('Shortage magnitude percentile',fontsize=20, fontname = 'Gill Sans MT')
    ax1.set_ylabel('Annual shortage (Million $m^3$)',fontsize=20, fontname = 'Gill Sans MT')

#Figure in metric units
fig, axes = plt.subplots(2,3)
for s in range(len(axes.flat)):
    ax = axes.flat[s]
    histData = np.loadtxt('../../'+design+'/Infofiles/' +  all_IDs[s] + '/' + all_IDs[s] + '_info_0.txt')[:,2]*1233.4818/1000000
    synthetic = np.zeros([len(histData), samples*realizations])
    for j in range(samples):
        data= np.loadtxt('../../'+design+'/Infofiles/' +  all_IDs[s] + '/' + all_IDs[s] + '_info_' + str(j+1) + '.txt') 
        try:
            synthetic[:,j*realizations:j*realizations+realizations]=data[:,idx]*1233.4818/1000000
        except IndexError:
            print(all_IDs[s] + '_info_' + str(j+1))
    plotSDC(ax, synthetic, histData)

fig.set_size_inches([20,10])
fig.savefig('./Paper1_figures/userimpacts_fig5_SI.svg')
fig.savefig('./Paper1_figures/userimpacts_fig5_SI.png')


    
