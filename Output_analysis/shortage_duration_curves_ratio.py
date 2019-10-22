import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches
from scipy import stats
import itertools
import os
from mpi4py import MPI
import math
plt.ioff()

WDs = ['36','37','38','39','45','50','51','52','53','70','72']
#non_irrigation_structures = np.genfromtxt('non_irrigation.txt',dtype='str').tolist() #list IDs of structures of interest
#irrigation_structures = [[]]*len(WDs) 
#for i in range(len(WDs)):
#    irrigation_structures[i] = np.genfromtxt(WDs[i]+'_irrigation.txt',dtype='str').tolist()
#irrigation_structures_flat = [item for sublist in irrigation_structures for item in sublist]
all_IDs = np.genfromtxt('./TBD.txt',dtype='str').tolist() #irrigation_structures_flat+WDs+non_irrigation_structures
nStructures = len(all_IDs)
percentiles = np.arange(0,100)
samples = 1000
realizations = 10

if not os.path.exists('./RatioShortageCurves/'):
    os.makedirs('./RatioShortageCurves/')

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
  
def plotSDC(synthetic_shortage, synthetic_demand, histData_shortage, histData_demand,uncurt_shortage, uncurt_demand, structure_name):
    n = 12
    #Reshape historic data to a [no. years x no. months] matrix
    f_hist_d = np.reshape(histData_demand, (int(np.size(histData_demand)/n), n))
    f_hist_s = np.reshape(histData_shortage, (int(np.size(histData_shortage)/n), n))
    f_uncurt_d = np.reshape(uncurt_demand, (int(np.size(uncurt_demand)/n), n))
    f_uncurt_s = np.reshape(uncurt_shortage, (int(np.size(uncurt_shortage)/n), n))
    #Reshape to annual totals
    f_hist_totals_ratio = np.sum(f_hist_s,1)/np.sum(f_hist_d,1)
    f_uncurt_totals_ratio = np.sum(f_uncurt_s,1)/np.sum(f_uncurt_d,1)
    #Calculate historical shortage duration curves
    F_hist = np.sort(f_hist_totals_ratio) # for inverse sorting add this at the end [::-1]
    F_uncurt = np.sort(f_uncurt_totals_ratio)
    
    #Reshape synthetic data
    #Create matrix of [no. years x no. months x no. samples]
    synthetic_global_s = np.zeros([int(np.size(histData_demand)/n),n,samples*realizations]) 
    synthetic_global_d = np.zeros([int(np.size(histData_demand)/n),n,samples*realizations]) 
    # Loop through every SOW and reshape to [no. years x no. months]
    for j in range(samples*realizations):
        synthetic_global_s[:,:,j]= np.reshape(synthetic_shortage[:,j], (int(np.size(synthetic_shortage[:,j])/n), n))
        synthetic_global_d[:,:,j]= np.reshape(synthetic_demand[:,j], (int(np.size(synthetic_demand[:,j])/n), n))
    #Reshape to annual totals
    synthetic_global_totals_ratio = np.sum(synthetic_global_s,1)/np.sum(synthetic_global_d,1)
#    multi_year_durations = [[]]*samples
#    
#    # Count consecutive years of shortage
#    for i in range(samples):
#        multi_year_durations[i] = shortage_duration(synthetic_global_totals[:,i])
#    hist_durations = shortage_duration(f_hist_totals)
#    
    p=np.arange(100,0,-10)
#    p_i=p[::-1]
#    hist_durations_percentiles = np.zeros([len(p_i)])
#    multi_year_durations_percentiles = np.zeros([len(p_i),samples])
#    for i in range(samples):
#        for j in range(len(p_i)):
#            if hist_durations:
#                hist_durations_percentiles[j] = np.percentile(hist_durations,p_i[j])
#            if multi_year_durations[i]:
#                multi_year_durations_percentiles[j,i] = np.percentile(multi_year_durations[i],p_i[j])
#    
#    fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
#    # ax1
#    handles = []
#    labels=[]
#    color = '#000292'
#    for i in range(len(p)):
#        ax1.fill_between(p_i, np.min(multi_year_durations_percentiles[:,:],1), np.percentile(multi_year_durations_percentiles[:,:], p[i], axis=1), color=color, alpha = 0.1)
#        ax1.plot(p_i, np.percentile(multi_year_durations_percentiles[:,:], p[i], axis=1), linewidth=0.5, color=color, alpha = 0.3)
#        handle = matplotlib.patches.Rectangle((0,0),1,1, color=color, alpha=alpha(i, base=0.1))
#        handles.append(handle)
#        label = "{:.0f} %".format(100-p[i])
#        labels.append(label)
#    ax1.plot(p_i,hist_durations_percentiles, c='black', linewidth=2, label='Historical record')
#    ax1.set_xlim(0,100)
#    ax1.legend(handles=handles, labels=labels, framealpha=1, fontsize=8, loc='upper left', title='Frequency in experiment',ncol=2)
#    ax1.set_xlabel('Shortage magnitude percentile', fontsize=12)
#    ax1.set_ylabel('Years of continuous shortages', fontsize=12)
#    fig.suptitle('Duration of shortage for ' + structure_name, fontsize=16)
#    fig.savefig('./MultiyearShortageCurves/' + structure_name + '.svg')
#    fig.savefig('./MultiyearShortageCurves/' + structure_name + '.png')
#    fig.clf()
    
    #Calculate synthetic shortage duration curves
    F_syn = np.empty([int(np.size(histData_shortage)/n),samples*realizations])
    F_syn[:] = np.NaN
    for j in range(samples*realizations):
        F_syn[:,j] = np.sort(synthetic_global_totals_ratio[:,j])
    
    # For each percentile of magnitude, calculate the percentile among the experiments ran
    perc_scores = np.zeros_like(F_syn) 
    for m in range(int(np.size(histData_shortage)/n)):
        perc_scores[m,:] = [stats.percentileofscore(F_syn[m,:], j, 'rank') for j in F_syn[m,:]]
                
    P = np.arange(1.,len(F_hist)+1)*100 / len(F_hist)
    
    
    ylimit = round(np.max(F_syn))
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
    ax1.plot(P,F_uncurt, c='black', linewidth=2, linestyle = '-', label='Historical record (uncurtailed)')
    ax1.set_ylim(0,ylimit)
    ax1.set_xlim(0,100)
    #ax1.legend(handles=handles, labels=labels, framealpha=1, fontsize=8, loc='upper left', title='Frequency in experiment',ncol=2)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_xlabel('Shortage ratio percentile', fontsize=12)
    ax1.set_ylabel('Ration of annual shortage to annual demand', fontsize=12)

    fig.suptitle('Shortage ratio for ' + structure_name, fontsize=16)
    plt.subplots_adjust(bottom=0.2)
    fig.savefig('./RatioShortageCurves/' + structure_name + '.svg')
    fig.savefig('./RatioShortageCurves/' + structure_name + '.png')
    fig.clf()
    
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
    if all_IDs[i] in WDs:
        histData_shortage = np.zeros(105*12) #105 years x 12 months
        histData_demand = np.zeros(105*12)
        synthetic_shortage = np.zeros([len(histData_shortage),samples, realizations])
        synthetic_demand = np.zeros([len(histData_shortage),samples, realizations])
        for ID in irrigation_structures[WDs.index(all_IDs[i])]:
            histData_shortage += np.loadtxt('./Infofiles/' +  ID + '/' + ID + '_info_0.txt')[:,2]
            histData_demand += np.loadtxt('./Infofiles/' +  ID + '/' + ID + '_info_0.txt')[:,1]
            for j in range(samples):
                for r in range(realizations):
                    data_s= np.loadtxt('./Infofiles/' +  ID + '/' + ID + '_info_' + str(j+1) + '_' + str(r+1) + '.txt')[:,2] 
                    data_d= np.loadtxt('./Infofiles/' +  ID + '/' + ID + '_info_' + str(j+1) + '_' + str(r+1) + '.txt')[:,1] 
                synthetic_shortage[:,j,r]+=data_s
                synthetic_demand[:,j,r]+=data_d
    else:
        histData_shortage = np.loadtxt('./Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_0.txt')[:,2]
        histData_demand = np.loadtxt('./Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_0.txt')[:,1]
        uncurt_shortage = np.loadtxt('./Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_0_uncurtailed.txt')[:,2]
        uncurt_demand = np.loadtxt('./Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_0_uncurtailed.txt')[:,1]
        synthetic_shortage = np.zeros([len(histData_shortage),samples, realizations])
        synthetic_demand = np.zeros([len(histData_shortage),samples, realizations])
        for j in range(samples):
            for r in range(realizations):
                data_s= np.loadtxt('./Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_' + str(j+1) + '_' + str(r+1) + '.txt')[:,2]     
                data_d= np.loadtxt('./Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_' + str(j+1) + '_' + str(r+1) + '.txt')[:,1]     
                synthetic_shortage[:,j,r]=data_s
                synthetic_demand[:,j,r]=data_d
    # Reshape into timeseries x all experiments
    synthetic_shortage = np.reshape(synthetic_shortage, (len(histData_shortage), samples*realizations))
    synthetic_demand = np.reshape(synthetic_demand, (len(histData_shortage), samples*realizations))
    plotSDC(synthetic_shortage, synthetic_demand, histData_shortage, histData_demand, uncurt_shortage, uncurt_demand, all_IDs[i])

    
