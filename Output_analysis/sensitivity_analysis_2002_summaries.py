import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import sys
plt.ioff()

design = str(sys.argv[1])
sensitive_output = str(sys.argv[2])

idx = np.arange(2,22,2)

LHsamples = np.loadtxt('../Qgen/' + design + '.txt') 

param_bounds=np.loadtxt('../Qgen/uncertain_params_'+design[10:-5]+'.txt', usecols=(1,2))
SOW_values = np.array([1,1,1,1,0,0,1,1,1,1,1,0,0,0]) #Default parameter values for base SOW
samples = len(LHsamples[:,0])
realizations = 10
params_no = len(LHsamples[0,:])
param_names=[x.split(' ')[0] for x in open('../Qgen/uncertain_params_'+design[10:-5]+'.txt').readlines()]
problem = {
    'num_vars': params_no,
    'names': param_names,
    'bounds': param_bounds.tolist()
}
percentiles = np.arange(0,100)
all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist() 
nStructures = len(all_IDs)

# deal with fact that calling result.summary() in statsmodels.api
# calls scipy.stats.chisqprob, which no longer exists
scipy.stats.chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)

#==============================================================================
# Function for water years
#==============================================================================
empty=[]
n=12

def sensitivity_analysis_per_structure(ID):
    '''
    Perform analysis for shortage frequency
    '''
    HIS_short = np.loadtxt('../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_0.txt')[:,2]

    # Reshape into water years
    # Create matrix of [no. years x no. months x no. experiments]
    f_HIS_short = np.reshape(HIS_short, (int(np.size(HIS_short)/n), n))

    # Shortage per water year
    f_HIS_short_WY = np.sum(f_HIS_short,axis=1)

    # Identify percentile for 2002 annual shortage
    percentile_2002=int(np.round(scipy.stats.percentileofscore(f_HIS_short_WY, f_HIS_short_WY[93], kind='strict'), decimals=0))
    
    S1 = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ ID + '_S1.csv')
    S1_conf = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ ID + '_S1_conf.csv')
    DELTA = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ ID + '_DELTA.csv')
    DELTA_conf = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ ID + '_DELTA_conf.csv')
    R2_scores = pd.read_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/'+ ID + '_R2.csv')
    return DELTA[str(percentile_2002)].values, DELTA_conf[str(percentile_2002)].values, S1[str(percentile_2002)].values, S1_conf[str(percentile_2002)].values, R2_scores[str(percentile_2002)].values

frequency_delta = pd.DataFrame(np.zeros((nStructures, params_no)), columns = param_names)
frequency_delta_conf = pd.DataFrame(np.zeros((nStructures, params_no)), columns = param_names)
frequency_S1 = pd.DataFrame(np.zeros((nStructures, params_no)), columns = param_names)
frequency_S1_conf = pd.DataFrame(np.zeros((nStructures, params_no)), columns = param_names)
frequency_R2 = pd.DataFrame(np.zeros((nStructures, params_no)), columns = param_names)
frequency_delta.index = frequency_delta_conf.index = frequency_S1.index = frequency_S1_conf.index = frequency_R2.index = all_IDs

for ID in all_IDs:
    frequency_delta.at[ID], frequency_delta_conf.at[ID],frequency_S1.at[ID], frequency_S1_conf.at[ID],frequency_R2.at[ID] = sensitivity_analysis_per_structure(ID)
    

frequency_delta.to_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/2002_DELTA.csv')
frequency_delta_conf.to_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/2002_DELTA_conf.csv')
frequency_S1.to_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/2002_S1.csv')
frequency_S1_conf.to_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/2002_S1_conf.csv')
frequency_R2.to_csv('../'+design+'/'+sensitive_output+'_Sensitivity_analysis/2002_R2.csv')