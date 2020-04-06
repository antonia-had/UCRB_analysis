import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import math
from mpi4py import MPI
import sys
plt.ioff()

design = str(sys.argv[1])
sensitive_output = str(sys.argv[2])

all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist()
nStructures = len(all_IDs)
# Longform parameter names to use in figure legend
parameter_names_long = ['IWR demand mutliplier', 'Reservoir loss', 
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

color_list = ["#F18670", "#E24D3F", "#CF233E", "#681E33", "#676572", "#F3BE22", "#59DEBA", "#14015C", "#DAF8A3", "#0B7A0A", "#F8FFA2", "#578DC0", "#4E4AD8", "#32B3F7","#F77632"] 
titles = ["Delta","S1","R2"]
yaxistitles = ["Change explained","Variance explained","Variance explained"]

def plotindeces(structure_name):    
    p=np.arange(100,-10,-10)

    delta_values = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_DELTA.csv')
    delta_CI_values = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_DELTA_conf.csv')
    delta_values.set_index(list(delta_values)[0],inplace=True)
    delta_CI_values.set_index(list(delta_CI_values)[0],inplace=True)
    delta_values = delta_values.clip(lower=0)
    delta_CI_values = delta_CI_values.clip(lower=0)
    
    S1_values = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_S1.csv')
    S1_CI_values = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_S1_conf.csv')
    S1_values.set_index(list(S1_values)[0],inplace=True)
    S1_CI_values.set_index(list(S1_CI_values)[0],inplace=True)
    S1_values = S1_values.clip(lower=0)
    S1_CI_values = S1_CI_values.clip(lower=0)
    for p in percentiles:
        total = np.sum(S1_values[str(p)])
        if total!=0 and total<1:
            diff = 1-total
            S1_values.at['Interaction',str(p)]=diff
        else:
            S1_values.at['Interaction',str(p)]=0               

    R2_values = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_R2.csv')
    R2_values.set_index(list(R2_values)[0],inplace=True)
    R2_values = R2_values.clip(lower=0)
    for p in percentiles:
        total = np.sum(R2_values[str(p)])
        if total!=0 and total<1:
            diff = 1-total
            R2_values.at['Interaction',str(p)]=diff
        else:
            R2_values.at['Interaction',str(p)]=0
                  
    values_to_plot = [delta_values.values, S1_values.values, R2_values.values]
    CI_to_plot = [delta_CI_values.values, S1_CI_values.values]

    for k in range(len(titles)-1):
        indices = values_to_plot[k]
        indices_CI = CI_to_plot[k]
        fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
        for i in range(len(param_names)):
            ax1.fill_between(np.arange(0,100), indices[i,:]-indices_CI[i,:], indices[i,:]+indices_CI[i,:], color = color_list[i], alpha = 0.5)
            ax1.plot(np.arange(0,100), indices[i,:], color = color_list[i], linewidth=2)
        ax1.set_title(titles[k])
        ax1.set_ylim(0,1)
        ax1.set_xlim(0,100)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.set_ylabel(yaxistitles[k], fontsize=16)
        ax1.set_xlabel('Shortage magnitude percentile', fontsize=16)
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.8])
        ax1.legend(handles, labels = parameter_names_long, fontsize=10, loc='lower center',ncol = 5, bbox_to_anchor=(0.5, -0.5))
        fig.suptitle('Shortage magnitude sensitivity for '+ structure_name, fontsize=16)
        fig.savefig('../'+design+'/'+sensitive_output+'SensitivityCurves/' + structure_name + '_'+titles[k]+'_indices.svg')
        fig.savefig('../'+design+'/'+sensitive_output+'SensitivityCurves/' + structure_name + '_'+titles[k]+'_indices.png')
        
    indices = values_to_plot[2]
    fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
    for i in range(len(param_names)):
        ax1.plot(np.arange(0,100), indices[i,:], color = color_list[i], linewidth=2)
    ax1.set_title(titles[2])
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,100)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_ylabel(yaxistitles[2], fontsize=16)
    ax1.set_xlabel('Shortage magnitude percentile', fontsize=16)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.8])
    ax1.legend(handles, labels = parameter_names_long, fontsize=10, loc='lower center',ncol = 5, bbox_to_anchor=(0.5, -0.5))
    fig.suptitle('Shortage magnitude sensitivity for '+ structure_name, fontsize=16)
    fig.savefig('../'+design+'/'+sensitive_output+'SensitivityCurves/' + structure_name + '_'+titles[2]+'_indices.svg')
    fig.savefig('../'+design+'/'+sensitive_output+'SensitivityCurves/' + structure_name + '_'+titles[2]+'_indices.png')


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
    plotindeces(all_IDs[i])

    
