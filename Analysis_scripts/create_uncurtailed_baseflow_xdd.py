import numpy as np
from string import Template

transbasin = np.genfromtxt('./../Structures_files/TBD.txt',dtype='str').tolist()

# For DDM
# split data on periods (splitting on spaces/tabs doesn't work because some columns are next to each other)
with open('./../Statemod_files/cm2015B.ddm','r') as f:
    all_split_data_DDM = [x.split('.') for x in f.readlines()]       
f.close()        
# get unsplit data to rewrite firstLine # of rows
with open('./../Statemod_files/cm2015B.ddm','r') as f:
    all_data_DDM = [x for x in f.readlines()]       
f.close() 
# Get uncurtailed demands
with open('./../Statemod_files/cm2015_export_max.stm','r') as f:
    diversions_uc = [x.split()[1:14] for x in f.readlines()[78:94]]       
f.close() 
uncurtailed = [x[0] for x in diversions_uc] # Get uncurtailed structures

# =============================================================================
# Define functions that generate each type of input file 
# =============================================================================
firstLine = 779
# Function for DDM files
def writenewDDM(structures):    
    allstructures = []
    for m in range(len(structures)):
        allstructures.extend(structures[m])      
    new_data = []
    for i in range(len(all_split_data_DDM)-firstLine):
        row_data = []
        # Split first 3 columns of row on space
        # This is because the first month is lumped together with the year and the ID when spliting on periods
        row_data.extend(all_split_data_DDM[i+firstLine][0].split())
        # If the structure is not in the ones we care about then do nothing
        if row_data[1] in structures[0]: #If the structure is transbasin (to uncurtail)   
            # apply multiplier to 1st month
            row_data[2] = str(int(float(diversions_uc[uncurtailed.index(row_data[1])][1])))
            # apply multipliers to rest of the columns
            for j in range(1,12):
                row_data.append(str(int(float(diversions_uc[uncurtailed.index(row_data[1])][j+1])))) 
        else:
            for j in range(len(all_split_data_DDM[i+firstLine])-2):
                row_data.append(str(int(float(all_split_data_DDM[i+firstLine][j+1]))))                      
        # append row of adjusted data
        new_data.append(row_data)                
    # write new data to file
    f = open('./../Statemod_files/'+ 'cm2015B.ddm'[0:-4] + '_uncurtailed' + 'cm2015B.ddm'[-4::],'w')
    # write firstLine # of rows as in initial file
    for i in range(firstLine):
        f.write(all_data_DDM[i])            
    for i in range(len(new_data)):
        # write year, ID and first month of adjusted data
        f.write(new_data[i][0] + ' ' + new_data[i][1] + (19-len(new_data[i][1])-len(new_data[i][2]))*' ' + new_data[i][2] + '.')
        # write all but last month of adjusted data
        for j in range(len(new_data[i])-4):
            f.write((7-len(new_data[i][j+3]))*' ' + new_data[i][j+3] + '.')                
        # write last month of adjusted data
        f.write((9-len(new_data[i][-1]))*' ' + new_data[i][-1] + '.' + '\n')            
    f.close()
    
    return None

writenewDDM([transbasin])