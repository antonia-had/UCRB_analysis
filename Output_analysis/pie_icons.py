import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
from scipy import stats
import pandas as pd
import matplotlib.patches

plt.ioff()
WDs = ['36','37','38','39','45','50','51','52','53','70','72']
non_irrigation_structures = np.genfromtxt('non_irrigation.txt',dtype='str').tolist() #list IDs of structures of interest
irrigation_structures = [[]]*len(WDs) 
for i in range(len(WDs)):
    irrigation_structures[i] = np.genfromtxt(WDs[i]+'_irrigation.txt',dtype='str').tolist()
irrigation_structures_flat = [item for sublist in irrigation_structures for item in sublist]
all_IDs = non_irrigation_structures+WDs+irrigation_structures_flat
nStructures = len(all_IDs)
percentiles = np.arange(0,110, 10)
#Experiment directories
experiments = ['Colorado_global_experiment']
  
def gen_icon(structure_name):
    if not os.path.exists('./Marker_Icons/'):
        os.makedirs('./Marker_Icons/')
    color_list = ["#F18670", "#E24D3F", "#CF233E", "#681E33", "#676572", "#F3BE22", "#59DEBA", "#14015C", "#DAF8A3", "#0B7A0A", "#F8FFA2", "#578DC0", "#4E4AD8", "#F77632"]  
    delta_values = pd.read_csv('./Colorado_global_experiment/Magnitude_Sensitivity_analysis/'+ structure_name + '_DELTA.csv')
    delta_values.set_index(list(delta_values)[0],inplace=True)
    delta_values = delta_values.clip(lower=0)              
    for p in percentiles:
        fig, ax = plt.subplots(figsize=(1,1))
        extra_row = pd.DataFrame(data=np.array([np.zeros(100)]), index= ['Other'], columns=list(delta_values.columns.values))
        delta_values = delta_values.append(extra_row)
        ax.pie(delta_values[str(p)], colors=color_list)
        fig.savefig('./Marker_Icons/' + structure_name + '_delta_' + str(p) + '.png',transparent=True)
        plt.close()
        
#        S1_values = pd.read_csv('./Colorado_global_experiment/Magnitude_Sensitivity_analysis/'+ structure_name + '_S1.csv')
#        S1_values.set_index(list(S1_values)[0],inplace=True)
#        S1_values = S1_values.clip(lower=0)
#        extra_row = pd.DataFrame(data=np.array([np.zeros(100)]), index= ['Other'], columns=list(S1_values.columns.values))
#        S1_values = S1_values.append(extra_row)
#        for p in percentiles:
#            total = np.sum(S1_values[str(p)])
#            if total!=0:
#                value = 1-total
#                S1_values.set_value('Other',str(p),value)
#        ax.pie(S1_values[str(p)], colors=color_list)
#        fig.savefig('./Marker_Icons/' + structure_name + '_S1_' + str(p) + '.png',transparent=True)
#        plt.close()
#    
#        R2_values = pd.read_csv('./Colorado_global_experiment/Magnitude_Sensitivity_analysis/'+ structure_name + '_R2.csv')
#        R2_values.set_index(list(R2_values)[0],inplace=True)
#        R2_values = R2_values.clip(lower=0)
#        extra_row = pd.DataFrame(data=np.array([np.zeros(100)]), index= ['Other'], columns=list(R2_values.columns.values))
#        R2_values = R2_values.append(extra_row)
#        for p in percentiles:
#            total = np.sum(R2_values[str(p)])
#            if total!=0:
#                value = 1-total
#                R2_values.set_value('Other',str(p),value)
#        ax.pie(R2_values[str(p)], colors=color_list)
#        fig.savefig('./Marker_Icons/' + structure_name + '_R2_' + str(p) + '.png',transparent=True)
#        plt.close()
    

empty=[] 
for i in range(len(all_IDs)):
    gen_icon(all_IDs[i])

    
