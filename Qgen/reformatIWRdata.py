import numpy as np

def writeNewFiles(filename, firstLine, numSites):
    # split data on periods
    with open(filename,'r') as f:
        all_split_data = [x.split('.') for x in f.readlines()]
        
    f.close()
    
    numYears = int((len(all_split_data)-firstLine)/numSites)
    MonthlyIWR = np.zeros([12*numYears,numSites])
    for i in range(numYears):
        for j in range(numSites):
            index = firstLine + i*numSites + j
            all_split_data[index][0] = all_split_data[index][0].split()[2]
            MonthlyIWR[i*12:(i+1)*12,j] = np.asfarray(all_split_data[index][0:12], float)
            
    np.savetxt('MonthlyIWR.csv',MonthlyIWR,fmt='%d',delimiter=',')
    
    # calculate annual flows
    AnnualIWR = np.zeros([numYears,numSites])
    for i in range(numYears):
        AnnualIWR[i,:] = np.sum(MonthlyIWR[i*12:(i+1)*12],0)
        
    np.savetxt('AnnualIWR.csv',AnnualIWR,fmt='%d',delimiter=',')
            
    return None

writeNewFiles('cm2015B.iwr', 463, 379)