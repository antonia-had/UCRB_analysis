import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import os
from scipy import stats
import pandas as pd
import matplotlib.patches
import itertools
from mpi4py import MPI
import math

plt.ioff()
samples = 1000
WDs = ['36','37','38','39','45','50','51','52','53','70','72']
WD_names = ['Blue River','Eagle River','Roaring Fork','Rifle/Elk/Parachute',
            'Divide','Muddy/Troublesome','U. Colorado/Fraser',
            'Piney/Cottonwood','N. Colorado','Roan Creek','L. Colorado']
non_irrigation_structures = np.genfromtxt('non_irrigation.txt',dtype='str').tolist() #list IDs of structures of interest
irrigation_structures = [[]]*len(WDs) 
for i in range(len(WDs)):
    irrigation_structures[i] = np.genfromtxt(WDs[i]+'_irrigation.txt',dtype='str').tolist()
irrigation_structures_flat = [item for sublist in irrigation_structures for item in sublist]
all_IDs = non_irrigation_structures+WDs+irrigation_structures_flat
nStructures = len(all_IDs)
parameter_names_long = ['Wet-to-wet state prob.', 'Dry-to-dry state prob.',
                   'Wet state sigma', 'Wet state mu', 'Dry state sigma',
                   'Dry state mu', 'Evaporation delta', 'Env. flow senior right',
                   'Shoshone active', 'M&I demand multiplier', 
                   'TBD demand multiplier', 'Reservoir loss', 'IWR demand mutliplier']
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'Shoshone','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11']
percentiles = np.arange(0,100)
#Experiment directories
experiments = ['Colorado_global_experiment','Colorado_streamflow_experiment']

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
    percentiles = np.arange(0,100)
    d_hist = np.asarray(shortage_duration(histData))
    if d_hist.size != 0:
        d_hist_percentiles = [np.percentile(d_hist,p) for p in percentiles]

    d_synth_percentiles = np.zeros([len(experiments),len(d_hist_percentiles),samples])
    for e in range(len(experiments)):
            for j in range(samples):
                d_synth = np.asarray(shortage_duration(synthetic[e,:,j]))
                if d_synth.size != 0:
                    d_synth_percentiles[e,:,j] = [np.percentile(d_synth,p) for p in percentiles]
                    
    p=np.arange(100,0,-10)
    ylimit = round(np.max(d_synth_percentiles))   
            
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14.5,8))
    # ax1
    handles1 = []
    labels1=[]
    color1 = '#000292'
    color2 = '#920003'
    for i in range(len(p)):
        ax1.fill_between(percentiles, np.min(d_synth_percentiles[1,:,:],1), np.percentile(d_synth_percentiles[1,:,:], p[i], axis=1), color=color1, alpha = 0.1)
        ax1.plot(percentiles, np.percentile(d_synth_percentiles[1,:,:], p[i], axis=1), linewidth=0.5, color=color1, alpha = 0.3)
        handle = matplotlib.patches.Rectangle((0,0),1,1, color=color1, alpha=alpha(i, base=0.1))
        handles1.append(handle)
        label = "{:.0f}th".format(p[i])
        labels1.append(label)
    ax1.plot(percentiles,d_hist_percentiles, c='black', linewidth=2, label='Historical record')
    ax1.set_ylim(0,ylimit)
    ax1.set_xlim(0,100)
    ax1.legend(handles=handles1, labels=labels1, framealpha=1, fontsize=8, loc='upper left', title='Percentile',ncol=2)
    ax1.set_title("Streamflow experiment")
    # ax2
    handles2 = []
    labels2=[]
    for i in range(len(p)):
        ax2.fill_between(percentiles, np.min(d_synth_percentiles[0,:,:],1), np.percentile(d_synth_percentiles[0,:,:], p[i], axis=1), color=color2, alpha = 0.2)
        ax2.plot(percentiles, np.percentile(d_synth_percentiles[0,:,:], p[i], axis=1), linewidth=0.5, color=color2, alpha = 0.5)
        handle = matplotlib.patches.Rectangle((0,0),1,1, color=color2, alpha=alpha(i, base=0.2))
        handles2.append(handle)
        label = "{:.0f}th".format(p[i])
        labels2.append(label)        
    ax2.plot(percentiles,d_hist_percentiles, c='black', linewidth=2, label='Historical record')
    ax2.set_ylim(0,ylimit)
    ax2.set_xlim(0,100)
    ax2.legend(handles=handles2, labels=labels2, framealpha=1, fontsize=8, loc='upper left', title='Percentile',ncol=2)
    ax2.set_title("Global enseble experiment")
    # ax3
    for i in range(len(p)):
        ax3.plot(percentiles, np.percentile(d_synth_percentiles[1,:,:], p[i], axis=1), linewidth=1.5, color=color1, alpha = 0.5)
        ax3.plot(percentiles, np.percentile(d_synth_percentiles[0,:,:], p[i], axis=1), linewidth=1.5, color=color2, alpha = 0.5)
    ax3.plot(percentiles,d_hist_percentiles, c='black', linewidth=2, label='Historical record')
    ax3.set_ylim(0,ylimit)
    ax3.set_xlim(0,100)
    ax1.set_ylabel('Shortage duration (months)', fontsize=12)
    ax2.set_xlabel('Shortage duration percentile', fontsize=12)
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(handles3, labels3, fontsize=10, loc='upper left')
    fig.suptitle('Shortage durations for ' + structure_name, fontsize=16)
    plt.subplots_adjust(bottom=0.2)
    if not os.path.exists('./DurationPercentileCurves/'):
        os.makedirs('./DurationPercentileCurves/')
    fig.savefig('./DurationPercentileCurves/' + structure_name + '_global.svg')
    fig.savefig('./DurationPercentileCurves/' + structure_name + '_global.png')
    fig.clf()

empty=[] 
for i in range(start,stop):
    if all_IDs[i] in WDs:
        histData = np.zeros(1260)
        synthetic = np.zeros([len(experiments),len(histData),samples])
        for e in range(len(experiments)):
            for ID in irrigation_structures[WDs.index(all_IDs[i])]:
                histData += np.loadtxt('./'+experiments[0]+'/Infofiles/' +  ID + '/' + ID + '_info_0.txt')[:,2]
                for j in range(samples-1):
                    file = WDs[WDs.index(all_IDs[i])] + ' ' + ID + ' ' + experiments[e] + ' sample ' + str(j)
                    try:
                        data= np.loadtxt('./'+experiments[e] +'/Infofiles/' +  ID + '/' + ID + '_info_' + str(j+1) + '.txt')[:,2]     
                        synthetic[e,:,j]+=data
                    except:
                        empty.append(file)
    else:
        histData = np.loadtxt('./'+experiments[0]+'/Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_0.txt')[:,2]
        synthetic = np.zeros([len(experiments),len(histData),samples])
        for e in range(len(experiments)):
            for j in range(samples-1):
                    file = all_IDs[i] + ' ' + experiments[e] + ' sample ' + str(j)
                    try:
                        data= np.loadtxt('./'+experiments[e] +'/Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_' + str(j+1) + '.txt')[:,2]     
                        synthetic[e,:,j]=data
                    except:
                        empty.append(empty)
    plotSDC(synthetic, histData, all_IDs[i])

    
