import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches
from scipy import stats
import pandas as pd
import math
from mpi4py import MPI
import sys
import itertools
plt.ioff()

design = str(sys.argv[1])


all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist()
nStructures = len(all_IDs)
# Longform parameter names to use in figure legend
parameter_names_long = ['Min','IWR demand mutliplier', 'Reservoir loss', 
                        'TBD demand multiplier', 'M&I demand multiplier', 
                        'Shoshone active', 'Env. flow senior right', 
                        'Evaporation delta', 'Dry state mu', 
                        'Dry state sigma', 'Wet state mu', 
                        'Wet state sigma', 'Dry-to-dry state prob.', 
                        'Wet-to-wet state prob.', 'Earlier snowmelt', 'Interaction']
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'ShoshoneDMND','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift']
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

def shortage_duration(sequence, threshold):
    cnt_shrt = [sequence[i]>threshold for i in range(len(sequence))] # Returns a list of True values when there's a shortage
    shrt_dur = [ sum( 1 for _ in group ) for key, group in itertools.groupby( cnt_shrt ) if key ] # Counts groups of True values
    return shrt_dur

def plotSDC(synthetic, histData, structure_name):
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
    
    p=np.arange(100,-10,-10)
    
    #Calculate synthetic shortage duration curves
    F_syn = np.empty([int(np.size(histData)/n),samples])
    F_syn[:] = np.NaN
    for j in range(samples):
        F_syn[:,j] = np.sort(synthetic_global_totals[:,j])
    
    # For each percentile of magnitude, calculate the percentile among the experiments ran
    perc_scores = np.zeros_like(F_syn) 
    for m in range(int(np.size(histData)/n)):
        perc_scores[m,:] = [stats.percentileofscore(F_syn[m,:], j, 'rank') for j in F_syn[m,:]]
                
    P = np.arange(1.,len(F_hist)+1)*100 / len(F_hist)
   
    ylimit = np.max(F_syn)
    fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
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
    ax1.set_xlabel('Shortage magnitude percentile', fontsize=20)
    ax1.set_ylabel('Annual shortage (Million $m^3$)', fontsize=20)

    fig.suptitle('Shortage magnitudes for ' + structure_name, fontsize=16)
    plt.subplots_adjust(bottom=0.2)
    fig.savefig('../'+design+'/ShortagePercentileCurves/' + structure_name + '_metric.svg')
    fig.savefig('../'+design+'/ShortagePercentileCurves/' + structure_name + '_metric.png')
    fig.clf()
  
#    '''
#    Sensitivity analysis plots normalized to 0-100
#    '''
#    
#    delta_values = pd.read_csv('../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_DELTA.csv')
#    delta_values.set_index(list(delta_values)[0],inplace=True)
#    delta_values = delta_values.clip(lower=0)
#    for p in percentiles:
#        total = np.sum(delta_values[str(p)])
#        if total!=0:
#            for param in param_names:
#                    value = 100*delta_values.at[param,str(p)]/total
#                    delta_values.set_value(param,str(p),value)
#    delta_values_to_plot = delta_values.values.tolist()
#    
#    S1_values = pd.read_csv('../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_S1.csv')
#    S1_values.set_index(list(S1_values)[0],inplace=True)
#    S1_values = S1_values.clip(lower=0)
#    for p in percentiles:
#        total = np.sum(S1_values[str(p)])
#        if total!=0 and total<1:
#            diff = 1-total
#            S1_values.at['Interaction',str(p)]=diff
#        else:
#            S1_values.at['Interaction',str(p)]=0               
#    for column in S1_values:
#        S1_values[column] = S1_values[column]*100
#    S1_values_to_plot = S1_values.values.tolist()
#
#    R2_values = pd.read_csv('../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_R2.csv')
#    R2_values.set_index(list(R2_values)[0],inplace=True)
#    R2_values = R2_values.clip(lower=0)
#    for p in percentiles:
#        total = np.sum(R2_values[str(p)])
#        if total!=0 and total<1:
#            diff = 1-total
#            R2_values.at['Interaction',str(p)]=diff
#        else:
#            R2_values.at['Interaction',str(p)]=0  
#    for column in R2_values:
#        R2_values[column] = R2_values[column]*100
#    R2_values_to_plot = R2_values.values.tolist()
#    
#    color_list = ["#F18670", "#E24D3F", "#CF233E", "#681E33", "#676572", "#F3BE22", "#59DEBA", "#14015C", "#DAF8A3", "#0B7A0A", "#F8FFA2", "#578DC0", "#4E4AD8", "#32B3F7","#F77632"]  
#                  
#    values_to_plot = [delta_values_to_plot, S1_values_to_plot, R2_values_to_plot]
#    titles = ["Delta","S1","R2"]
#    yaxistitles = ["Change explained","Variance explained","Variance explained"]
#    for k in range(len(titles)):
#        fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
#        ax1.stackplot(np.arange(0,100), values_to_plot[k], colors = color_list)
#        ax1.set_title(titles[k])
#        ax1.set_ylim(0,150)
#        ax1.set_xlim(0,100)
#        handles, labels = ax1.get_legend_handles_labels()
#        ax1.set_ylabel(yaxistitles[k], fontsize=12)
#        ax1.set_xlabel('Shortage magnitude percentile', fontsize=12)
#        plt.legend(handles[1:], labels = parameter_names_long[1:], fontsize=10, loc='lower center',ncol = 5)
#        fig.suptitle('Shortage magnitude sensitivity for '+ structure_name, fontsize=16)
#        fig.savefig('../'+design+'/ShortageSensitivityCurves/' + structure_name + '_'+titles[k]+'_norm.svg')
#        fig.savefig('../'+design+'/ShortageSensitivityCurves/' + structure_name + '_'+titles[k]+'_norm.png')
    
    '''
    Sensitivity analysis plots
    '''
#    globalmax = [np.percentile(np.max(F_syn[:,:],1),p) for p in percentiles]
#    globalmin = [np.percentile(np.min(F_syn[:,:],1),p) for p in percentiles]
#    
#    delta_values = pd.read_csv('../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_DELTA.csv')
#    delta_values.set_index(list(delta_values)[0],inplace=True)
#    delta_values = delta_values.clip(lower=0)
#    bottom_row = pd.DataFrame(data=np.array([np.zeros(100)]), index= ['Interaction'], columns=list(delta_values.columns.values))
#    top_row = pd.DataFrame(data=np.array([globalmin]), index= ['Min'], columns=list(delta_values.columns.values))
#    delta_values = pd.concat([top_row,delta_values.loc[:],bottom_row])
#    for p in percentiles:
#        total = np.sum(delta_values[str(p)])-delta_values.at['Min',str(p)]
#        if total!=0:
#            for param in param_names:
#                    value = (globalmax[p]-globalmin[p])*delta_values.at[param,str(p)]/total
#                    delta_values.set_value(param,str(p),value)
#    for column in delta_values:
#        delta_values[column] = delta_values[column].round(decimals = 2)
#    delta_values_to_plot = delta_values.values.tolist()
#    
#    S1_values = pd.read_csv('../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_S1.csv')
#    S1_values.set_index(list(S1_values)[0],inplace=True)
#    S1_values = S1_values.clip(lower=0)
#    bottom_row = pd.DataFrame(data=np.array([np.zeros(100)]), index= ['Interaction'], columns=list(S1_values.columns.values))
#    top_row = pd.DataFrame(data=np.array([globalmin]), index= ['Min'], columns=list(S1_values.columns.values))
#    S1_values = pd.concat([top_row,S1_values.loc[:],bottom_row])
#    for p in percentiles:
#        total = np.sum(S1_values[str(p)])-S1_values.at['Min',str(p)]
#        if total!=0:
#            diff = 1-total
#            S1_values.set_value('Interaction',str(p),diff)
#            for param in param_names+['Interaction']:
#                value = (globalmax[p]-globalmin[p])*S1_values.at[param,str(p)]
#                S1_values.set_value(param,str(p),value)                
#    for column in S1_values:
#        S1_values[column] = S1_values[column].round(decimals = 2)
#    S1_values_to_plot = S1_values.values.tolist()
#
#    R2_values = pd.read_csv('../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_R2.csv')
#    R2_values.set_index(list(R2_values)[0],inplace=True)
#    R2_values = R2_values.clip(lower=0)
#    bottom_row = pd.DataFrame(data=np.array([np.zeros(100)]), index= ['Interaction'], columns=list(R2_values.columns.values))
#    top_row = pd.DataFrame(data=np.array([globalmin]), index= ['Min'], columns=list(R2_values.columns.values))
#    R2_values = pd.concat([top_row,R2_values.loc[:],bottom_row])
#    for p in percentiles:
#        total = np.sum(R2_values[str(p)])-R2_values.at['Min',str(p)]
#        if total!=0:
#            diff = 1-total
#            R2_values.set_value('Interaction',str(p),diff)
#            for param in param_names+['Interaction']:
#                value = (globalmax[p]-globalmin[p])*R2_values.at[param,str(percentiles[p])]
#                R2_values.set_value(param,str(percentiles[p]),value)
#    for column in R2_values:
#        R2_values[column] = R2_values[column].round(decimals = 2)
#    R2_values_to_plot = R2_values.values.tolist()
#    
#    color_list = ["white", "#F18670", "#E24D3F", "#CF233E", "#681E33", "#676572", "#F3BE22", "#59DEBA", "#14015C", "#DAF8A3", "#0B7A0A", "#F8FFA2", "#578DC0", "#4E4AD8", "#32B3F7","#F77632"]  
#                  
#    values_to_plot = [delta_values_to_plot, S1_values_to_plot, R2_values_to_plot]
#    titles = ["Delta","S1","R2"]
#    for k in range(len(titles)):
#        fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
#        ax1.stackplot(np.arange(0,100), values_to_plot[k], colors = color_list)
#        ax1.plot(percentiles, globalmax, color='black', linewidth=1)
#        ax1.plot(percentiles, globalmin, color='black', linewidth=1)
#        ax1.set_title(titles[k])
#        ax1.set_ylim(0,ylimit)
#        ax1.set_xlim(0,100)
#        handles, labels = ax1.get_legend_handles_labels()
#        ax1.set_ylabel('Annual shortage (af)', fontsize=12)
#        ax1.set_xlabel('Shortage magnitude percentile', fontsize=12)
#        plt.legend(handles[1:], labels = parameter_names_long[1:], fontsize=10, loc='lower center',ncol = 5)
#        plt.subplots_adjust(bottom=0.2)
#        fig.suptitle('Shortage magnitude sensitivity for '+ structure_name, fontsize=16)
#        fig.savefig('../'+design+'/ShortageSensitivityCurves/' + structure_name + '_'+titles[k]+'.svg')
#        fig.savefig('../'+design+'/ShortageSensitivityCurves/' + structure_name + '_'+titles[k]+'.png')

# Begin parallel simulation
comm = MPI.COMM_WORLD

# Get the number of processors and the rank of processors
rank = comm.rank
nprocs = comm.size

# Determine the chunk which each processor will neeed to do
count = int(math.floor(nStructures/nprocs))
remainder = nStructures % nprocs

# Use the processor rank to determine the chunk of work each processor will do
if rank < remainder:
	start = rank*(count+1)
	stop = start + count + 1
else:
	start = remainder*(count+1) + (rank-remainder)*count
	stop = start + count
    
for i in range(start, stop):
    histData = np.loadtxt('../'+design+'/Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_0.txt')[:,2]*1233.4818/1000000
    synthetic = np.zeros([len(histData), samples*realizations])
    for j in range(samples):
        data= np.loadtxt('../'+design+'/Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_' + str(j+1) + '.txt') 
        try:
            synthetic[:,j*realizations:j*realizations+realizations]=data[:,idx]*1233.4818/1000000
        except IndexError:
            print(all_IDs[i] + '_info_' + str(j+1))
    plotSDC(synthetic, histData, all_IDs[i])

    
