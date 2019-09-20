import numpy as np
from string import Template
import os
import pandas as pd

# =============================================================================
# Experiment set up
# =============================================================================

# Read/define relevant structures for each uncertainty
transbasin = np.genfromtxt('TBD.txt',dtype='str').tolist()

# =============================================================================
# Load global information (applicable to all SOW)
# =============================================================================
# For RSP
T = open('./Experiment_files/cm2015B_template.rsp', 'r')
template_RSP = Template(T.read())

# For DDM
# split data on periods (splitting on spaces/tabs doesn't work because some columns are next to each other)
with open('./Experiment_files/cm2015B.ddm','r') as f:
    all_split_data_DDM = [x.split('.') for x in f.readlines()]   
f.close()   
with open('./Experiment_files/cm2015B.ddm','r') as f:
    # get unsplit data to rewrite firstLine # of rows
    all_data_DDM = [x for x in f.readlines()]         
f.close() 
# Get uncurtailed demands
max_values = pd.DataFrame(np.zeros([6,13]),index=transbasin)
for i in range(len(all_split_data_DDM)-779):
    row_data = []
    row_data.extend(all_split_data_DDM[i+779][0].split())
    if row_data[1] in transbasin:
        current_values = max_values.loc[row_data[1]].values
        if float(row_data[2])>current_values[0]:
            current_values[0] = float(row_data[2])
        for j in range(len(all_split_data_DDM[i+779])-3):
            if float(all_split_data_DDM[i+779][j+1])>current_values[j+1]:
                current_values[j+1]=float(all_split_data_DDM[i+779][j+1])
        max_values.loc[row_data[1]]=current_values

for index, row in max_values.iterrows():
    row[12] = row.values[:-1].sum()
    
max_values.to_csv('maxvalues.csv')

# =============================================================================
# Define functions that generate each type of input file 
# =============================================================================

# Function for DDM files
def writenewDDM(structures, firstLine):      
    new_data = []
    for i in range(len(all_split_data_DDM)-firstLine):
        row_data = []
        # Split first 3 columns of row on space
        # This is because the first month is lumped together with the year and the ID when spliting on periods
        row_data.extend(all_split_data_DDM[i+firstLine][0].split())
        if row_data[1] in structures: #If the structure is transbasin (to uncurtail)   
            # apply multiplier to 1st month
            row_data[2] = str(int(max_values.loc[row_data[1]][0]))
            # apply multipliers to rest of the columns
            for j in range(1,13):
                row_data.append(str(int(max_values.loc[row_data[1]][j])))  
        elif row_data[1] not in structures:
            for j in range(len(all_split_data_DDM[i+firstLine])-2):
                row_data.append(str(int(float(all_split_data_DDM[i+firstLine][j+1]))))                      
        # append row of adjusted data
        new_data.append(row_data)                
    # write new data to file
    f = open('./Experiment_files/'+ 'cm2015B.ddm'[0:-4] + '_uncurt' + 'cm2015B.ddm'[-4::],'w')
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
 
d = {}
d['IWR'] = 'cm2015B.iwr'
d['XBM'] = 'cm2015x.xbm'
d['DDR'] = 'cm2015B.ddr'
d['DDM'] = 'cm2015B_uncurt.ddm'
d['EVA'] = 'cm2015.eva'
d['RES'] = 'cm2015B.res'
S1 = template_RSP.safe_substitute(d)
f1 = open('./Experiment_files/cm2015B_uncurt.rsp', 'w')
f1.write(S1)    
f1.close()
writenewDDM(transbasin, 779)
#os.system("./Experiment_files/statemod .Experiment_files/cm2015B_uncurt -simulate")