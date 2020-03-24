import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import sys
plt.ioff()

design = str(sys.argv[1])
sensitive_output = str(sys.argv[2])

transbasin = np.genfromtxt('../../Structures_files/TBD.txt',dtype='str').tolist()
all_IDs = ['3600687', '7000550', '7200799', '7200645', '3704614', '7202003']
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
percentiles = np.arange(0,101)
samples = 1000
realizations = 10
idx = np.arange(2,22,2)

color_list = ["#ff8000", "#b15a29", "#693c99", "#ffff98", "#680c0e", "#a8cfe5", "#fcbd6d", "#e2171a", "#f99998", "#32a02c", "#b2df8a", "#1b77b3", "#104162", "#1b5718","#cbb3d7"]
              
titles = ["Delta","S1","R2"]
yaxistitles = ["Change explained","Variance explained","Variance explained"]
  
def getdata(structure_name):    
    p=np.arange(100,-10,-10)
    '''
    Sensitivity analysis plots normalized to 0-100
    '''
    delta_values = pd.read_csv('../../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_DELTA.csv')
    delta_conf = pd.read_csv('../../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_DELTA_conf.csv')
    delta_values.set_index(list(delta_values)[0],inplace=True)
    delta_conf.set_index(list(delta_conf)[0],inplace=True)
    delta_values = delta_values.clip(lower=0)
    for p in percentiles:
        # Check if their CI overlaps zero or if are lower than the dummy
        for param in param_names:
            if delta_values.at[param,str(p)]<delta_conf.at[param,str(p)] or \
            delta_values.at[param,str(p)]<=delta_values.at['Controlvariable',str(p)]:
                # If yes, set the index value to zero
                delta_values.set_value(param,str(p),0)
    delta_values=delta_values.drop(['Controlvariable'])
    for p in percentiles:           
        total = np.sum(delta_values[str(p)])
        if total!=0:
            for param in param_names:
                    value = 100*delta_values.at[param,str(p)]/total
                    delta_values.set_value(param,str(p),value)
    delta_values_to_plot = delta_values.values.tolist()
    
    S1_values = pd.read_csv('../../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_S1.csv')
    S1_conf = pd.read_csv('../../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_S1_conf.csv')
    S1_values.set_index(list(S1_values)[0],inplace=True)
    S1_conf.set_index(list(S1_conf)[0],inplace=True)
    S1_values = S1_values.clip(lower=0)
    for p in percentiles:
        # Check if their CI overlaps zero or if are lower than the dummy
        for param in param_names:
            if S1_values.at[param,str(p)]<S1_values.at[param,str(p)] or \
            S1_values.at[param,str(p)]<=S1_values.at['Controlvariable',str(p)]:
                # If yes, set the index value to zero
                S1_values.set_value(param,str(p),0)
    S1_values=S1_values.drop(['Controlvariable'])
    for p in percentiles:
        total = np.sum(S1_values[str(p)])
        if total!=0 and total<1:
            diff = 1-total
            S1_values.at['Interaction',str(p)]=diff
        else:
            S1_values.at['Interaction',str(p)]=0               
    for column in S1_values:
        S1_values[column] = S1_values[column]*100
    S1_values_to_plot = S1_values.values.tolist()

    R2_values = pd.read_csv('../../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ structure_name + '_R2.csv')
    R2_values.set_index(list(R2_values)[0],inplace=True)
    R2_values = R2_values.clip(lower=0)
    for p in percentiles:
        # Check if they are lower than the dummy
        for param in param_names:
            if R2_values.at[param,str(p)]<=R2_values.at['Controlvariable',str(p)]:
                # If yes, set the index value to zero
                R2_values.set_value(param,str(p),0)
    R2_values=R2_values.drop(['Controlvariable'])
    for p in percentiles:
        total = np.sum(R2_values[str(p)])
        if total!=0 and total<1:
            diff = 1-total
            R2_values.at['Interaction',str(p)]=diff
        else:
            R2_values.at['Interaction',str(p)]=0  
    for column in R2_values:
        R2_values[column] = R2_values[column]*100
    R2_values_to_plot = R2_values.values.tolist()
    
    return [delta_values_to_plot, S1_values_to_plot, R2_values_to_plot]
  
fig, axes = plt.subplots(3,3, figsize=(14.5,8))
for i in range(len(all_IDs[:3])):
    values_to_plot = getdata(all_IDs[:3][i])
    for j in range(len(values_to_plot)):
        ax = axes.flat[i+j*3]
        ax.stackplot(np.arange(0,100), values_to_plot[j], colors = color_list)
        ax.set_title(titles[j])
        ax.set_ylim(0,100)
        ax.set_xlim(0,100)
        if i==0 and j==0:
            handles, labels = ax.get_legend_handles_labels()
        if i==0:
            ax.set_ylabel(yaxistitles[j], fontsize=12)
        if j==2:
            ax.set_xlabel('Shortage magnitude percentile', fontsize=12)
plt.legend(handles[1:], labels = parameter_names_long[1:], fontsize=10, loc='lower center',ncol = 5)
fig.suptitle(sensitive_output + ' sensitivity', fontsize=16)
fig.savefig('./Paper2_figures/'+sensitive_output+'_3IWR_norm.svg')
fig.savefig('./Paper2_figures/'+sensitive_output+'_3IWR_norm.png')

fig, axes = plt.subplots(3,3, figsize=(14.5,8))
for i in range(len(all_IDs[3:])):
    values_to_plot = getdata(all_IDs[3:][i])
    for j in range(len(values_to_plot)):
        ax = axes.flat[i+j*3]
        ax.stackplot(np.arange(0,100), values_to_plot[j], colors = color_list)
        ax.set_title(titles[j])
        ax.set_ylim(0,100)
        ax.set_xlim(0,100)
        if i==0 and j==0:
            handles, labels = ax.get_legend_handles_labels()
        if i==0:
            ax.set_ylabel(yaxistitles[j], fontsize=12)
        if j==2:
            ax.set_xlabel('Shortage magnitude percentile', fontsize=12)
plt.legend(handles[1:], labels = parameter_names_long[1:], fontsize=10, loc='lower center',ncol = 5)
fig.suptitle(sensitive_output + ' sensitivity', fontsize=16)
fig.savefig('./Paper2_figures/'+sensitive_output+'_3others_norm.svg')
fig.savefig('./Paper2_figures/'+sensitive_output+'_3others_norm.png')

    
