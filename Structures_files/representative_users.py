import math
import numpy as np
import pandas as pd

# Read/define relevant structures for each uncertainty
transbasin = np.genfromtxt('../Structures_files/TBD.txt',dtype='str').tolist()
irrigation = np.genfromtxt('../Structures_files/irrigation.txt',dtype='str').tolist()
env_flows = ['7202003']

rights = pd.read_csv('../Structures_files/diversions_admin.csv')

'''Find most senior irrigation structures
'''
rights = rights.sort_values(by=['Admin'])

count=0
for index, row in rights.iterrows():
    if str(row['WDID']) in irrigation:
        print('Senior irrigation structure: '+ str(row['WDID']))
        count+=1
    if count==3:
        break
    
'''Find most junior irrigation structures
'''    
rights = rights.sort_values(by=['Admin'], ascending=False)

count=0
for index, row in rights.iterrows():
    if row['Admin']<99999 and str(row['WDID']) in irrigation:
        print('Junior irrigation structure: '+ str(row['WDID']))
        count+=1
    if count==3:
        break

''' Find median irrigation structures
''' 
mediandecree=np.median(rights['Decree'].values)
medianadmin=np.median(rights['Admin'].values)

'''Find irrigation structure with highest decree
'''
rights = rights.sort_values(by=['Decree'], ascending=False)

count=0
for index, row in rights.iterrows():
    if row['Admin']<99999 and str(row['WDID']) in irrigation:
        print('Irrigation structure with highest decree: '+ str(row['WDID']))
        count+=1
    if count==3:
        break
    
'''Find TBD structure with highest decree
'''
rights = rights.sort_values(by=['Decree'], ascending=False)

for index, row in rights.iterrows():
    if row['Admin']<99999 and str(row['WDID']) in transbasin:
        print('Transbasin structure with highest decree: '+ str(row['WDID']))
        break
    
'''Find TBD structure with lowest decree
'''
rights = rights.sort_values(by=['Decree'])

for index, row in rights.iterrows():
    if row['Admin']<99999 and str(row['WDID']) in transbasin:
        print('Transbasin structure with highest decree: '+ str(row['WDID']))
        break