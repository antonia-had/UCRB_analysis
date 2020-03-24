import numpy as np
import pandas as pd
import sys
from mpi4py import MPI
sys.path.append('../')
from SALib.analyze import delta
import statsmodels.api as sm
import scipy.stats
import math
from os import path


design = str(sys.argv[1])
LHsamples = np.loadtxt('../Qgen/' + design + '.txt') 
# convert streamflow multipliers/deltas in LHsamples to HMM parameter values
def convertMultToParams(multipliers):
    params = np.zeros(np.shape(multipliers))
    params[:,0] = multipliers[:,0]*15.258112 # historical dry state mean
    params[:,1] = multipliers[:,1]*0.259061 # historical dry state std dev
    params[:,2] = multipliers[:,2]*15.661007 # historical wet state mean
    params[:,3] = multipliers[:,3]*0.252174 # historical wet state std dev
    params[:,4] = multipliers[:,4] + 0.679107 # historical dry-dry transition prob
    params[:,5] = multipliers[:,5] + 0.649169 # historical wet-wet transition prob
    
    return params

# convert HMM parameter values to streamflow multipliers/deltas in LHsamples 
def convertParamsToMult(params):
    multipliers = np.zeros(np.shape(params))
    multipliers[:,0] = params[:,0]/15.258112 # historical dry state mean
    multipliers[:,1] = params[:,1]/0.259061 # historical dry state std dev
    multipliers[:,2] = params[:,2]/15.661007 # historical wet state mean
    multipliers[:,3] = params[:,3]/0.252174 # historical wet state std dev
    multipliers[:,4] = params[:,4] - 0.679107 # historical dry-dry transition prob
    multipliers[:,5] = params[:,5] - 0.649169 # historical wet-wet transition prob
    
    return multipliers

# find SOWs where mu0 > mu1 and swap their wet and dry state parameters
HMMparams = convertMultToParams(LHsamples[:,[7,8,9,10,11,12]])
for i in range(np.shape(HMMparams)[0]):
    if HMMparams[i,0] > HMMparams[i,2]: # dry state mean above wet state mean
        # swap dry and wet state parameters to correct labels
        mu0 = HMMparams[i,2]
        std0 = HMMparams[i,3]
        mu1 = HMMparams[i,0]
        std1 = HMMparams[i,1]
        p00 = HMMparams[i,5]
        p11 = HMMparams[i,4]
        newParams = np.array([[mu0, std0, mu1, std1, p00, p11]])
        LHsamples[i,[7,8,9,10,11,12]] = convertParamsToMult(newParams)

# Add dummy control variable
LHsamples = np.concatenate((LHsamples, np.random.rand(1000,1)), axis=1)
param_bounds=np.loadtxt('../Qgen/uncertain_params_'+design[10:-5]+'.txt', usecols=(1,2))
# Add dummy control variable bounds
param_bounds = np.concatenate((param_bounds, [[0,1]]))

SOW_values = np.array([1,1,1,1,0,0,1,1,1,1,1,0,0,0]) #Default parameter values for base SOW
samples = len(LHsamples[:,0])
realizations = 10
params_no = len(LHsamples[0,:])

rows_to_keep = np.intersect1d(np.where(LHsamples[:,0]>=0)[0],np.where(LHsamples[:,0]<=0)[0])
for i in range(params_no):
    within_rows = np.intersect1d(np.where(LHsamples[:,i] > param_bounds[i][0])[0], np.where(LHsamples[:,i] < param_bounds[i][1])[0])
    rows_to_keep = np.union1d(rows_to_keep,within_rows)
LHsamples = LHsamples[rows_to_keep,:]

param_names=[x.split(' ')[0] for x in open('../Qgen/uncertain_params_'+design[10:-5]+'.txt').readlines()]+['Controlvariable']

problem = {
    'num_vars': params_no,
    'names': param_names,
    'bounds': param_bounds.tolist()
}
all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist() 
nStructures = len(all_IDs)

# deal with fact that calling result.summary() in statsmodels.api
# calls scipy.stats.chisqprob, which no longer exists
scipy.stats.chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)

def fitOLS(dta, predictors):
    # concatenate intercept column of 1s
    dta['Intercept'] = np.ones(np.shape(dta)[0])
    # get columns of predictors
    cols = dta.columns.tolist()[-1:] + predictors
    #fit OLS regression
    ols = sm.OLS(dta['Shortage'], dta[cols])
    result = ols.fit()
    return result

def sensitivity_analysis_per_structure(ID):
#    if not path.exists('../'+design+'/Max_Duration_Sensitivity_analysis/'+ ID + '_DELTA_conf.csv'):
    '''
    Perform analysis for max shortage duration at each magnitude
    '''
    durations = np.load('../'+design+'/Max_Duration_Curves/' + ID + '.npy')
    dta = pd.DataFrame(data = np.repeat(LHsamples, realizations, axis = 0), columns=param_names)
    DELTA = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = percentiles)
    DELTA_conf = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = percentiles)
    S1 = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = percentiles)
    S1_conf = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = percentiles)
    R2_scores = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = percentiles)
    DELTA.index=DELTA_conf.index=S1.index=S1_conf.index = R2_scores.index = param_names
    for i in range(len(percentiles)):
        if durations[i].any():
#            # Delta Method analysis
#            try:
#                output = np.mean(durations[i].reshape(-1, 10), axis=1)
#                result= delta.analyze(problem, LHsamples, output, print_to_console=False)
#                DELTA[percentiles[i]]= result['delta']
#                DELTA_conf[percentiles[i]] = result['delta_conf']
#                S1[percentiles[i]]=result['S1']
#                S1_conf[percentiles[i]]=result['S1_conf']
#            except:
#                pass
            # OLS regression analysis
            dta['Shortage']=durations[i]
            try:
                for m in range(params_no):
                    predictors = dta.columns.tolist()[m:(m+1)]
                    result = fitOLS(dta, predictors)
                    R2_scores.at[param_names[m],percentiles[i]]=result.rsquared
            except:
                pass          
    R2_scores.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ ID + '_R2.csv')
#    S1.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ ID + '_S1.csv')
#    S1_conf.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ ID + '_S1_conf.csv')
#    DELTA.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ ID + '_DELTA.csv')
#    DELTA_conf.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ ID + '_DELTA_conf.csv')
        
# =============================================================================
# Start parallelization (running each structure in parallel)
# =============================================================================
    
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

# Run simulation
for k in range(start, stop):
    sensitivity_analysis_per_structure(all_IDs[k])