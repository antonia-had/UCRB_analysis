import numpy as np

all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist()
demands = np.zeros([len(all_IDs), 1260])
shortages = np.zeros([len(all_IDs), 1260])

for i in range(len(all_IDs)):
    print(all_IDs[i])
    histData = np.loadtxt('../LHsamples_original_1000/Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_0.txt')
    demands[i,:] = histData[:,1]
    shortages[i,:] = histData[:,2]

np.savetxt('../Summary_info/demands_uncurtailed.csv', demands, delimiter=",")
np.savetxt('../Summary_info/shortages_uncurtailed.csv', shortages, delimiter=",")