from mpi4py import MPI
import math
import numpy as np

# =============================================================================
# Experiment set up
# =============================================================================

# Read in SOW parameters
LHsamples = np.loadtxt('./LHsamples_wider.txt') 
realizations = 10

# Read/define relevant structures for each uncertainty
ID = '7202003'

# =============================================================================
# Define output extraction function
# =============================================================================

lines=[]
with open ('./Infofiles_wide/' +  ID + '/' + ID + '_streamflow_0.txt','w') as f:
    with open ('./cm2015B.xdd', 'rt') as xdd_file:
        for line in xdd_file.readlines()[39:]:
                data = line.split('.')
                if len(data)>1:
                    introdata = data[0].split()
                    if introdata[0]==ID:
                        if introdata[3]!='TOT':
                            lines.append([introdata[2], data[24]])
    xdd_file.close()
    for line in lines:
        for item in line:
            f.write("%s\t" % item)
        f.write("\n")
f.close()

def getinfo(s):
    lines=[]
    with open ('./Infofiles_wide/' +  ID + '/' + ID + '_streamflow_' + str(s+1) +'.txt','w') as f:
        with open ('./Experiment_files/cm2015B_S'+ str(s+1)+ '_1.xdd', 'rt') as xdd_file:
            test = xdd_file.readline()
            if test:
                for line in xdd_file.readlines()[39:]:
                    data = line.split('.')
                    if len(data)>1:
                        introdata = data[0].split()
                        if introdata[0]==ID:
                            if introdata[3]!='TOT':
                                lines.append([introdata[2], data[24]])
        xdd_file.close()
        for j in range(1, realizations):
            count=0
            try:
                with open ('./Experiment_files/cm2015B_S'+ str(s+1)+ '_' + str(j+1) + '.xdd', 'rt') as xdd_file:
                    test = xdd_file.readline()
                    if test:
                        for line in xdd_file.readlines()[39:]:
                            data = line.split('.')
                            if len(data)>1:
                                introdata = data[0].split()
                                if introdata[0]==ID:
                                    if introdata[3]!='TOT':
                                        lines[count].append(data[24])
                                        count+=1
                    else:
                        for i in range(len(lines)):
                            lines[i].extend('-999')
                xdd_file.close()
            except IOError:
                for i in range(len(lines)):
                    lines[i].extend('-999')
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
count = int(math.floor(len(LHsamples[:,0])/nprocs))
remainder = len(LHsamples[:,0]) % nprocs

# Use the processor rank to determine the chunk of work each processor will do
if rank < remainder:
	start = rank*(count+1)
	stop = start + count + 1
else:
	start = remainder*(count+1) + (rank-remainder)*count
	stop = start + count
    
for s in range(start, stop):
        getinfo(s)