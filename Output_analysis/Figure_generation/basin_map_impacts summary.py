import numpy as np
import cartopy.feature as cpf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
import pandas as pd
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

extent = [-109.069,-105.6,38.85,40.50]
extent_large = [-111.0,-101.0,36.5,41.5]
#rivers_10m = cpf.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m')
tiles = cimgt.StamenTerrain(style='terrain')
shape_feature = ShapelyFeature(Reader('../../Structures_files/Shapefiles/Water_Districts.shp').geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='None')
flow_feature = ShapelyFeature(Reader('../../Structures_files/Shapefiles/UCRBstreams.shp').geometries(), ccrs.PlateCarree(), edgecolor='royalblue', facecolor='None')

months = 12
years = 105
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'ShoshoneDMND','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift']
color_list = ["#F18670", "#E24D3F", "#CF233E", "#681E33", "#676572", "#F3BE22", "#59DEBA", "#14015C", "#DAF8A3", "#0B7A0A", "#F8FFA2", "#578DC0", "#4E4AD8", "#32B3F7","#F77632"]  
        
structures = pd.read_csv('../../Structures_files/modeled_diversions.csv',index_col=0)
all_IDs = np.genfromtxt('../../Structures_files/metrics_structures_old.txt',dtype='str').tolist()
demands = pd.read_csv('../../Summary_info/demands_uncurtailed.csv',header = None, index_col=False)
shortages = pd.read_csv('../../Summary_info/shortages_uncurtailed.csv',header = None, index_col=False)
demands['index']=all_IDs
demands = demands.set_index('index')
shortages['index']=all_IDs
shortages = shortages.set_index('index')

demands = demands.loc[~demands.index.duplicated(keep='first')]
shortages = shortages.loc[~shortages.index.duplicated(keep='first')]

yearly_demands = np.sum(np.reshape(demands.values, (len(demands), years, months)), axis=2)
yearly_shortages = np.sum(np.reshape(shortages.values, (len(demands), years, months)), axis=2)

rights = pd.read_csv('../../Structures_files/diversions_admin.csv')
groupbyadmin = rights.groupby('WDID')
right_groups_admin = groupbyadmin.apply(lambda x: list(x['Admin']))
right_groups_decree = groupbyadmin.apply(lambda x: list(x['Decree']))
right_groups_source = groupbyadmin.apply(lambda x: x['Source'].unique())
agg_rights = pd.DataFrame(np.zeros([len(right_groups_admin),3]), 
                          index = right_groups_admin.index,
                          columns=['Admin', 'Decree','Source'])
for i in range(len(right_groups_admin)):
    weights=right_groups_decree[i] 
    if np.sum(weights)>0:
        entries = [np.average(right_groups_admin[i],weights=weights)]
    else:
        entries = [99999.99999]
    entries.extend((np.sum(right_groups_decree[i]), right_groups_source[i][0]))
    agg_rights.at[right_groups_admin.index[i]] = entries
agg_rights = agg_rights.reindex(list(demands.index))
structures = structures.reindex(list(demands.index))
agg_rights['WD'] = [int(ID[:2]) for ID in list(agg_rights.index)]
wdcounts = agg_rights['WD'].value_counts().sort_index()

indices = pd.read_csv('2002_DELTA_max_duration.csv',index_col=0)
indices['1stFactor'] = indices.idxmax(axis=1)
indices = indices.reindex(list(demands.index))
colors = [color_list[param_names.index(f)] for f in indices['1stFactor'].values]

ratio2002 = np.divide(yearly_shortages[:,93],yearly_demands[:,93],
                      out=np.zeros_like(yearly_shortages[:,93]), where=yearly_demands[:,93]!=0)

fig = plt.figure(figsize=(18, 9))
ax = plt.axes(projection=tiles.crs)
#ax.add_feature(rivers_10m, facecolor='None', edgecolor='royalblue', linewidth=2, zorder=4)
ax.add_image(tiles, 9, interpolation='none',alpha = 0.8)
ax.set_extent(extent)
ax.add_feature(shape_feature, facecolor='#a1a384',alpha = 0.6)
ax.add_feature(flow_feature, alpha = 0.6, linewidth=1.5, zorder=4)              
points = ax.scatter(structures['X'], structures['Y'], marker = '.', s = ratio2002*400, c = colors ,transform=ccrs.Geodetic(), zorder=5)

#geom = geometry.box(extent[0],extent[2], extent[1], extent[3])    
#fig = plt.figure(figsize=(18, 9))
#ax = plt.axes(projection=tiles.crs)
#ax.add_feature(rivers_10m, facecolor='None', edgecolor='royalblue', linewidth=2, zorder=4)
#ax.add_image(tiles, 7, interpolation='none',alpha = 0.8)
#ax.set_extent(extent_large)
#ax.add_feature(cpf.STATES)
#ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor='None', edgecolor='black', linewidth=2,)
#ax.add_feature(shape_feature, facecolor='#a1a384',alpha = 0.6)
