import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import matplotlib.pyplot as plt
import math
import sys
from mpi4py import MPI
sys.path.append('../')
from multiprocessing import Pool
from SALib.analyze import delta
plt.ioff()

design = str(sys.argv[1])
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
all_IDs = np.genfromtxt('../Structures_files/irrigation.txt',dtype='str').tolist()
nStructures = len(all_IDs)

# deal with fact that calling result.summary() in statsmodels.api
# calls scipy.stats.chisqprob, which no longer exists
scipy.stats.chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)

#==============================================================================
# Function for water years
#==============================================================================
empty=[]
n=12
HIS_short = np.loadtxt('../'+design+'/Infofiles/7202003/7202003_info_1.txt')[:,2]

def shortages_per_structure(ID):
    SYN_short = np.zeros([len(HIS_short), samples * realizations])
    for j in range(samples):
        data= np.loadtxt('../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_' + str(j+1) + '.txt') 
        try:
            SYN_short[:,j*realizations:j*realizations+realizations]=data[:,idx]
        except IndexError:
            print(ID + '_info_' + str(j+1))
    # Reshape into water years
    # Create matrix of [no. years x no. months x no. experiments]
    f_SYN_short = np.zeros([int(np.size(HIS_short)/n),n, samples*realizations])
    for i in range(samples*realizations):
        f_SYN_short[:,:,i]= np.reshape(SYN_short[:,i], (int(np.size(SYN_short[:,i])/n), n))

    # Shortage per water year
    f_SYN_short_WY = np.sum(f_SYN_short,axis=1)
    print (ID)
    return(f_SYN_short_WY)

# =============================================================================
# Start parallelization
# =============================================================================
    
# Begin parallel simulation
comm = MPI.COMM_WORLD

# Get the number of processors and the rank of processors
rank = comm.rank
nprocs = comm.size

p = Pool(nprocs)

allshortages = p.map(shortages_per_structure, all_IDs)

irrigationshortage = np.sum(allshortages, axis=0)
np.savetxt('../'+design+'/Infofiles/totalirrigationshortages.txt', irrigationshortage)

