import numpy as np
import cartopy.feature as cpf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
import pandas as pd
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

demands = pd.read_csv('../Summary_info/demands.csv',index_col=0)
structures = pd.read_csv('../Structures_files/modeled_diversions.csv',index_col=0)

for index, row in structures.iterrows():
    structures.at[str(index),'Mean demand'] = np.mean(demands.loc[str(index)].values)

extent = [-109.069,-105.6,38.85,40.50]
extent_large = [-111.0,-101.0,36.5,41.5]
#rivers_10m = cpf.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m')
tiles = cimgt.StamenTerrain(style='terrain')
shape_feature = ShapelyFeature(Reader('../Structures_files/Shapefiles/Water_Districts.shp').geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='None')
flow_feature = ShapelyFeature(Reader('../Structures_files/Shapefiles/UCRBstreams.shp').geometries(), ccrs.PlateCarree(), edgecolor='royalblue', facecolor='None')


fig = plt.figure(figsize=(18, 9))
ax = plt.axes(projection=tiles.crs)
#ax.add_feature(rivers_10m, facecolor='None', edgecolor='royalblue', linewidth=2, zorder=4)
ax.add_image(tiles, 9, interpolation='none',alpha = 0.8)
ax.set_extent(extent)
ax.add_feature(shape_feature, facecolor='#a1a384',alpha = 0.6)
ax.add_feature(flow_feature, alpha = 0.6, linewidth=1.5, zorder=4)              
points = ax.scatter(structures['X'], structures['Y'], marker = '.', s = 50, c = 'black' ,transform=ccrs.Geodetic(), zorder=5)

#geom = geometry.box(extent[0],extent[2], extent[1], extent[3])    
#fig = plt.figure(figsize=(18, 9))
#ax = plt.axes(projection=tiles.crs)
#ax.add_feature(rivers_10m, facecolor='None', edgecolor='royalblue', linewidth=2, zorder=4)
#ax.add_image(tiles, 7, interpolation='none',alpha = 0.8)
#ax.set_extent(extent_large)
#ax.add_feature(cpf.STATES)
#ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor='None', edgecolor='black', linewidth=2,)
#ax.add_feature(shape_feature, facecolor='#a1a384',alpha = 0.6)
