import numpy as np
import cartopy.feature as cpf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
import pandas as pd
import matplotlib.patches as mpatches
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from shapely import geometry


structures = pd.read_csv('modeled_diversions.csv',index_col=0)
specialstructures = ['7202003', '5300584', '5104634', '5104655', '3604684', 
                     '3804625SU', '3804617', '3704614']

extent = [-109.069,-105.6,38.85,40.50]
extent_large = [-111.0,-101.0,36.5,41.5]
rivers_10m = cpf.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m')
tiles = cimgt.StamenTerrain(style='terrain')
shape_feature = ShapelyFeature(Reader('Water_Districts.shp').geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='None')

'''
Plot all diversions per timestep
'''
#fig = plt.figure(figsize=(18, 9))
#ax = plt.axes(projection=tiles.crs)
#ax.add_feature(rivers_10m, facecolor='None', edgecolor='royalblue', linewidth=2, zorder=4)
#ax.add_image(tiles, 9, interpolation='none',alpha = 0.8)
#ax.set_extent(extent)
#ax.add_feature(shape_feature, facecolor='#a1a384',alpha = 0.6)
#points = ax.scatter(structures['X'], structures['Y'], marker = '.', s = 40, c = '#465241' ,transform=ccrs.Geodetic(), zorder=5)
#for ID in specialstructures: 
#    ax.scatter(structures.at[ID, 'X'], structures.at[ID, 'Y'], marker = '.', s = 80, c = 'black' ,transform=ccrs.Geodetic(), zorder=5)
geom = geometry.box(extent[0],extent[2], extent[1], extent[3])    
fig = plt.figure(figsize=(18, 9))
ax = plt.axes(projection=tiles.crs)
ax.add_feature(rivers_10m, facecolor='None', edgecolor='royalblue', linewidth=2, zorder=4)
ax.add_image(tiles, 7, interpolation='none',alpha = 0.8)
ax.set_extent(extent_large)
ax.add_feature(cpf.STATES)
ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor='None', edgecolor='black', linewidth=2,)
ax.add_feature(shape_feature, facecolor='#a1a384',alpha = 0.6)
