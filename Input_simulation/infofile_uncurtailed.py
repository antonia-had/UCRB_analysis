from mpi4py import MPI
import math
import numpy as np
import os

# =============================================================================
# Experiment set up
# =============================================================================

# Read in SOW parameters
#LHsamples = np.loadtxt('./LHsamples_wider.txt') 
#nSamples = len(LHsamples[:,0])
LHsamples = [299, 591, 754, 599, 765, 949]
nSamples = len(LHsamples)
realizations = 10

# Read/define relevant structures for each uncertainty
reservoirs = np.genfromtxt('reservoirs.txt',dtype='str').tolist()
transbasin = np.genfromtxt('TBD.txt',dtype='str').tolist()
irrigation = np.genfromtxt('irrigation.txt',dtype='str').tolist()
mun_ind = np.genfromtxt('M_I.txt',dtype='str').tolist()
env_flows = ['7202003']
shoshone = ['5300584']

# List IDs of structures of interest for output files
IDs = np.genfromtxt('metrics_structures.txt',dtype='str').tolist()
info_clmn = [2, 4, 17] # Define columns of aspect of interest 

# =============================================================================
# Define output extraction function
# =============================================================================
  
def getinfo(k):
    ID=IDs[k]
    if not os.path.exists('./Infofiles_wide/' + ID):
        os.makedirs('./Infofiles_wide/' + ID)
    for s in LHsamples:#range(nSamples):
        lines=[]
        with open ('./Infofiles_wide/' +  ID + '/' + ID + '_info_' + str(s+1) +'.txt','w') as f:
            with open ('./Experiment_files/cm2015B_S'+ str(s+1)+ '_1.xdd', 'rt') as xdd_file:
                for line in xdd_file:
                    data = line.split()
                    if data:
                        if data[0]==ID:
                            if data[3]!='TOT':
                                lines.append([data[2], data[4], data[17]])
            xdd_file.close()
            for j in range(1, realizations):
                count=0
                try:
                    with open ('./Experiment_files/cm2015B_S'+ str(s+1)+ '_' + str(j+1) + '.xdd', 'rt') as xdd_file:
                        test = xdd_file.readline()
                        if test:
                            for line in xdd_file:
                                data = line.split()
                                if data:
                                    if data[0]==ID:
                                        if data[3]!='TOT':
                                            lines[count].extend([data[4], data[17]])
                                            count+=1
                        else:
                            for i in range(len(lines)):
                                lines[i].extend(['-999.','-999.'])
                    xdd_file.close()
                except IOError:
                    for i in range(len(lines)):
                        lines[i].extend(['-999.','-999.'])
            for line in lines:
                for item in line:
                    f.write("%s\t" % item)
                f.write("\n")
        f.close()

# =============================================================================
# Start parallelization
# =============================================================================
    
# Begin parallel simulation
comm = MPI.COMM_WORLD

# Get the number of processors and the rank of processors
rank = comm.rank
nprocs = comm.size

# Determine the chunk which each processor will neeed to do
count = int(math.floor(len(IDs)/nprocs))
remainder = len(IDs) % nprocs

# Use the processor rank to determine the chunk of work each processor will do
if rank < remainder:
	start = rank*(count+1)
	stop = start + count + 1
else:
	start = remainder*(count+1) + (rank-remainder)*count
	stop = start + count
    
for k in range(start, stop):
        getinfo(k)

#comm.Barrier()
#if rank==1:        
#    os.system("rm ./Experiment_files/cm2015B_S*.xdd ./Experiment_files/cm2015B_S*.xre ./Experiment_files/cm2015B_S*.xss ./Experiment_files/cm2015B_S*.b44 ./Experiment_files/cm2015B_S*.b67 ./Experiment_files/cm2015B_S*.b43")