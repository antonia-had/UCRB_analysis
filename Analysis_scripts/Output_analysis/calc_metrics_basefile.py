import numpy as np
import os

def getinfo(ID):
    line_out = '' #Empty line for storing data to print in file   
    # Get summarizing files for each structure and aspect of interest from the .xdd or .xss files
    if not os.path.exists('./Infofiles/' + ID):
        os.makedirs('./Infofiles/' + ID) 
    with open ('./Infofiles/' +  ID + '/' + ID + '_info_0.txt','w') as f:
        try:
            with open ('./cm2015B.xdd', 'rt') as xdd_file:
                for line in xdd_file:
                    data = line.split()
                    if data:
                        if data[0]==ID:
                            if data[3]!='TOT':
                                for o in [2, 4, 17]:
                                    line_out+=(data[o]+'\t')
                                f.write(line_out)
                                f.write('\n')
                                line_out = ''
            xdd_file.close()
            f.close()
        except IOError:
            f.write('999999\t999999\t999999')
            f.close()


IDs = np.genfromtxt('metrics_structures_short.txt',dtype='str').tolist() #list IDs of structures of interest 
for ID in IDs:
    getinfo(ID)
    
nsamples = 1000

WDs = ['36','37','38','39','45','50','51','52','53','70','72'] 
irrigation_structures = [[]]*len(WDs) 
for i in range(len(WDs)):
    irrigation_structures[i] = np.genfromtxt(WDs[i]+'_irrigation.txt',dtype='str').tolist()
    
years = np.loadtxt('./Infofiles/7202003/7202003_info_0.txt',usecols = (0))
for i in range(len(WDs)):    
    if not os.path.exists('./Infofiles/' + WDs[i]):        
        os.makedirs('./Infofiles/' + WDs[i])    
    for j in range(nsamples+1):        
        accum = np.zeros([1260,2])        
        for ID in irrigation_structures[i]:            
            try:                
                accum += np.loadtxt('./Infofiles/' +  ID + '/' + ID + '_info_' + str(j) +'.txt',usecols = (1,2))            
            except:                
                accum +=np.zeros([1260,2])        
        np.savetxt('./Infofiles/' + WDs[i] + '/' + WDs[i] + '_info_' + str(j) +'.txt',np.concatenate((years[:, np.newaxis],accum), axis=1))