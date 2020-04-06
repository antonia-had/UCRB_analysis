import numpy as np

def writeNewFiles(filename, firstLine, sampleNo, realizationNo, allMonthlyData, version, design):
    nSites = np.shape(allMonthlyData)[1]
    
    # split data on periods
    with open('../Statemod_files/' + filename,'r') as f:
        all_split_data = [x.split('.') for x in f.readlines()]
        
    f.close()
        
    # get unsplit data to rewrite firstLine # of rows
    with open('../Statemod_files/' + filename,'r') as f:
        all_data = [x for x in f.readlines()]
        
    f.close()
    
    # replace former flows with new flows
    new_data = []
    for i in range(len(all_split_data)-firstLine):
        year_idx = int(np.floor(i/(nSites)))
        #print(year_idx)
        site_idx = np.mod(i,(nSites))
        #print(site_idx)
        row_data = []
        # split first 3 columns of row on space and find 1st month's flow
        row_data.extend(all_split_data[i+firstLine][0].split())
        row_data[2] = str(int(allMonthlyData[year_idx,site_idx,0]))
        # find remaining months' flows
        for j in range(11):
            row_data.append(str(int(allMonthlyData[year_idx,site_idx,j+1])))
            
        # find total flow
        row_data.append(str(int(np.sum(allMonthlyData[year_idx,site_idx,:]))))
            
        # append row of adjusted data
        new_data.append(row_data)

    f = open('../'+design+'/Experiment_files/'+ filename[0:-4] + '_S' + str(sampleNo) + '_' + str(realizationNo) + version + filename[-4::],'w')
    # write firstLine # of rows as in initial file
    for i in range(firstLine):
        f.write(all_data[i])
        
    for i in range(len(new_data)):
        # write year, ID and first month of adjusted data
        f.write(new_data[i][0] + ' ' + new_data[i][1] + (19-len(new_data[i][1])-len(new_data[i][2]))*' ' + new_data[i][2] + '.')
        # write all but last month of adjusted data
        for j in range(len(new_data[i])-4):
            f.write((7-len(new_data[i][j+3]))*' ' + new_data[i][j+3] + '.')
            
        # write last month of adjusted data
        if filename[-4::] == '.xbm':
            if len(new_data[i][-1]) <= 7:
                f.write((7-len(new_data[i][-1]))*' ' + new_data[i][-1] + '.' + '\n')
            else:
                f.write('********\n')
        else:
            f.write((9-len(new_data[i][-1]))*' ' + new_data[i][-1] + '.' + '\n')
        
    f.close()
    
    return None
