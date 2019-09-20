import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches
from scipy import stats
#import pandas as pd
import itertools
#import os
from mpi4py import MPI
import math
plt.ioff()

transbasin = np.genfromtxt('TBD.txt',dtype='str').tolist()
#WDs = ['36','37','38','39','45','50','51','52','53','70','72']
#non_irrigation_structures = np.genfromtxt('non_irrigation.txt',dtype='str').tolist() #list IDs of structures of interest
#irrigation_structures = [[]]*len(WDs) 
#for i in range(len(WDs)):
#    irrigation_structures[i] = np.genfromtxt(WDs[i]+'_irrigation.txt',dtype='str').tolist()
#irrigation_structures_flat = [item for sublist in irrigation_structures for item in sublist]
all_IDs = np.genfromtxt('./metrics_structures.txt',dtype='str').tolist()[:2] #irrigation_structures_flat+WDs+non_irrigation_structures
nStructures = len(transbasin)
# Longform parameter names to use in figure legend
#parameter_names_long = ['Min','IWR demand mutliplier', 'Reservoir loss', 
#                        'TBD demand multiplier', 'M&I demand multiplier', 
#                        'Shoshone active', 'Env. flow senior right', 
#                        'Evaporation delta', 'Dry state mu', 
#                        'Dry state sigma', 'Wet state mu', 
#                        'Wet state sigma', 'Dry-to-dry state prob.', 
#                        'Wet-to-wet state prob.', 'Earlier snowmelt', 'Interaction']
#param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
#             'ShoshoneDMND','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
#             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift']
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
    synthetic_global = np.zeros([int(np.size(histData)/n),n,samples]) 
    # Loop through every SOW and reshape to [no. years x no. months]
    for j in range(samples):
        synthetic_global[:,:,j]= np.reshape(synthetic[:,j], (int(np.size(synthetic[:,j])/n), n))
    #Reshape to annual totals
    synthetic_global_totals = np.sum(synthetic_global,1) 
    multi_year_durations = [[]]*samples
    
    # Count consecutive years of shortage
    for i in range(samples):
        multi_year_durations[i] = shortage_duration(synthetic_global_totals[:,i])
    hist_durations = shortage_duration(f_hist_totals)
    
    p=np.arange(100,-10,-10)
    p_i=p[::-1]
    hist_durations_percentiles = np.zeros([len(p_i)])
    multi_year_durations_percentiles = np.zeros([len(p_i),samples])
    for i in range(samples):
        for j in range(len(p_i)):
            if hist_durations:
                hist_durations_percentiles[j] = np.percentile(hist_durations,p_i[j])
            if multi_year_durations[i]:
                multi_year_durations_percentiles[j,i] = np.percentile(multi_year_durations[i],p_i[j])
    
    fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
    # ax1
    handles = []
    labels=[]
    color = '#000292'
    for i in range(len(p)):
        ax1.fill_between(p_i, np.min(multi_year_durations_percentiles[:,:],1), np.percentile(multi_year_durations_percentiles[:,:], p[i], axis=1), color=color, alpha = 0.1)
        ax1.plot(p_i, np.percentile(multi_year_durations_percentiles[:,:], p[i], axis=1), linewidth=0.5, color=color, alpha = 0.3)
        handle = matplotlib.patches.Rectangle((0,0),1,1, color=color, alpha=alpha(i, base=0.1))
        handles.append(handle)
        label = "{:.0f} %".format(100-p[i])
        labels.append(label)
    ax1.plot(p_i,hist_durations_percentiles, c='black', linewidth=2, label='Historical record')
    ax1.set_xlim(0,100)
    ax1.legend(handles=handles, labels=labels, framealpha=1, fontsize=8, loc='upper left', title='Frequency in experiment',ncol=2)
    ax1.set_xlabel('Shortage magnitude percentile', fontsize=12)
    ax1.set_ylabel('Years of continuous shortages', fontsize=12)
    fig.suptitle('Duration of shortage for ' + structure_name, fontsize=16)
    fig.savefig('./MultiyearShortageCurves_wide/' + structure_name + '.svg')
    fig.savefig('./MultiyearShortageCurves_wide/' + structure_name + '.png')
    fig.clf()
    
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
    
    
    ylimit = round(np.max(F_syn), -3)
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
    ax1.set_xlabel('Shortage magnitude percentile', fontsize=12)
    ax1.set_ylabel('Annual shortage (af)', fontsize=12)

    fig.suptitle('Shortage magnitudes for ' + structure_name, fontsize=16)
    plt.subplots_adjust(bottom=0.2)
    fig.savefig('./ShortagePercentileCurves_wide/' + structure_name + '.svg')
    fig.savefig('./ShortagePercentileCurves_wide/' + structure_name + '.png')
    fig.clf()
    
#    '''
#    Sensitivity analysis plots
#    '''
#    globalmax = [np.percentile(np.max(F_syn[:,:],1),p) for p in percentiles]
#    globalmin = [np.percentile(np.min(F_syn[:,:],1),p) for p in percentiles]
#    
#    delta_values = pd.read_csv('./Magnitude_Sensitivity_analysis/'+ structure_name + '_DELTA.csv')
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
    
#    S1_values = pd.read_csv('./Magnitude_Sensitivity_analysis/'+ structure_name + '_S1.csv')
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
#    R2_values = pd.read_csv('./Magnitude_Sensitivity_analysis/'+ structure_name + '_R2.csv')
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
    
#    color_list = ["white", "#F18670", "#E24D3F", "#CF233E", "#681E33", "#676572", "#F3BE22", "#59DEBA", "#14015C", "#DAF8A3", "#0B7A0A", "#F8FFA2", "#578DC0", "#4E4AD8", "#32B3F7","#F77632"]     
#    
#    #fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14.5,8))
#    fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
#    ax1.stackplot(np.arange(0,100), delta_values_to_plot, colors = color_list)
#    ax1.plot(percentiles, globalmax, color='black', linewidth=1)
#    ax1.plot(percentiles, globalmin, color='black', linewidth=1)
#    ax1.set_title("Delta index")
#    ax1.set_ylim(0,ylimit)
#    ax1.set_xlim(0,100)
#    ax2.stackplot(np.arange(0,100), S1_values_to_plot, colors = color_list)
#    ax2.plot(percentiles, globalmax, color='black', linewidth=1)
#    ax2.plot(percentiles, globalmin, color='black', linewidth=1)
#    ax2.set_title("S1")
#    ax2.set_ylim(0,ylimit)
#    ax2.set_xlim(0,100)
#    ax3.stackplot(np.arange(0,100), R2_values_to_plot, colors = color_list)
#    ax3.plot(percentiles, globalmax, color='black', linewidth=1)
#    ax3.plot(percentiles, globalmin, color='black', linewidth=1)
#    ax3.set_title("R^2")
#    ax3.set_ylim(0,ylimit)
#    ax3.set_xlim(0,100)
#    handles, labels = ax1.get_legend_handles_labels()
#    ax1.set_ylabel('Annual shortage (af)', fontsize=12)
#    ax1.set_xlabel('Shortage magnitude percentile', fontsize=12)
#    plt.legend(handles[1:], labels = parameter_names_long[1:], fontsize=10, loc='lower center',ncol = 5)
#    plt.subplots_adjust(bottom=0.2)
#    fig.suptitle('Shortage magnitude sensitivity for '+ structure_name, fontsize=16)
#    fig.savefig('./ShortageSensitivityCurves/' + structure_name + '_delta.svg')
#    fig.savefig('./ShortageSensitivityCurves/' + structure_name + '_delta.png')
    
'''
Get data for historic file.
'''
def getinfo(ID):
    line_out = '' #Empty line for storing data to print in file   
    # Get summarizing files for each structure and aspect of interest from the .xdd or .xss files
    with open ('./Infofiles_wide/' +  ID + '/' + ID + '_info_0.txt','w') as f:
        try:
            if ID in transbasin:
                with open ('./Experiment_files/cm2015B_S_uncurtailed.xdd', 'rt') as xdd_file:
                    for line in xdd_file:
                        data = line.split()
                        if data:
                            if data[0]==ID:
                                if data[3]!='TOT':
                                    for o in [2, 4, 17]:
                                        line_out+=(data[o]+'\t')
                                    f.write(line_out)
                                    f.write('\n')
                                    line_out = ''
                xdd_file.close()
                f.close()
            else:
                with open ('./cm2015B.xdd', 'rt') as xdd_file:
                    for line in xdd_file:
                        data = line.split()
                        if data:
                            if data[0]==ID:
                                if data[3]!='TOT':
                                    for o in [2, 4, 17]:
                                        line_out+=(data[o]+'\t')
                                    f.write(line_out)
                                    f.write('\n')
                                    line_out = ''
                xdd_file.close()
                f.close()
        except IOError:
            f.write('999999\t999999\t999999')
            f.close()

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
    getinfo(all_IDs[i])
    getinfo(transbasin[i])
    
for i in range(start, stop):
#    if all_IDs[i] in WDs:
#        histData = np.zeros(105*12) #105 years x 12 months
#        synthetic = np.zeros([len(histData),samples, realizations])
#        # if ID is a water disctrict
#        for ID in irrigation_structures[WDs.index(all_IDs[i])]:
#            histData += np.loadtxt('./Infofiles/' +  ID + '/' + ID + '_info_0.txt')[:,2]
#            for j in range(samples):
#                for r in range(realizations):
#                    data= np.loadtxt('./Infofiles/' +  ID + '/' + ID + '_info_' + str(j+1) + '_' + str(r+1) + '.txt')[:,2]     
#                synthetic[:,j,r]+=data
#    else:
    histData = np.loadtxt('./Infofiles_wide/' +  transbasin[i] + '/' + transbasin[i] + '_info_0.txt')[:,2]
    synthetic = np.zeros([len(histData), samples*realizations])
    for j in range(samples):
        data= np.loadtxt('./Infofiles_wide/' +  transbasin[i] + '/' + transbasin[i] + '_info_' + str(j+1) + '.txt') 
        try:
            synthetic[:,j*realizations:j*realizations+realizations]=data[:,idx]
        except IndexError:
            print(transbasin[i] + '_info_' + str(j+1))
    plotSDC(synthetic, histData, transbasin[i])

    
