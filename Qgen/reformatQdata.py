import numpy as np

def writeNewFiles(filename, firstLine, numSites):
    # split data on periods
    with open(filename,'r') as f:
        all_split_data = [x.split('.') for x in f.readlines()]
        
    f.close()
    
    numYears = int((len(all_split_data)-firstLine)/numSites)
    MonthlyQ = np.zeros([12*numYears,numSites])
    for i in range(numYears):
        for j in range(numSites):
            index = firstLine + i*numSites + j
            all_split_data[index][0] = all_split_data[index][0].split()[2]
            MonthlyQ[i*12:(i+1)*12,j] = np.asfarray(all_split_data[index][0:12], float)
            
    np.savetxt('MonthlyQ.csv',MonthlyQ,fmt='%d',delimiter=',')
    
    # calculate annual flows
    AnnualQ = np.zeros([numYears,numSites])
    for i in range(numYears):
        AnnualQ[i,:] = np.sum(MonthlyQ[i*12:(i+1)*12],0)
        
    np.savetxt('AnnualQ.csv',AnnualQ,fmt='%d',delimiter=',')
            
    return None

writeNewFiles('cm2015x.xbm',16,208)
