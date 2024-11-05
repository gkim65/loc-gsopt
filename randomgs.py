"""
Get n # of random points in a specific geographic region with associated gap times
"""

from src.sat_gen import satellites_from_constellation
from src.station_gen import gs_json, rand_gs_on_land,return_bdm_gs
from src.utils import load_earth_data, compute_all_gaps_contacts

from scipy.optimize import minimize
import numpy as np

import matplotlib.pyplot as plt

# Brahe Imports
import brahe as bh
import brahe.data_models as bdm
import brahe.access.access as ba

# random points function
import random_land_points as rlp



############################### SETUP: ###############################

# Make sure to load in earth inertial data every start time!
load_earth_data('data/iau2000A_finals_ab.txt')

############################### STEP 1: Satellites ###############################

# Loading in a satellite constellation 
CONSTELLATIONS = sorted(['YAM', 'UMBRA', 'SKYSAT', 'ICEYE', 'FLOCK', 'HAWK', 'CAPELLA', 'LEGION', 'WORLDVIEW', 'GEOEYE',
                  'NUSAT'])

constellation = CONSTELLATIONS[3]

satellites = satellites_from_constellation(constellation)

############################### STEP 2: Ground station generation ###############################

stations = gs_json('data/groundstations/atlas.json')

gs, counter, _,_ = rand_gs_on_land()

print(counter)

############################### STEP 3: Scenario Generation ###############################
# Get a random point on land in Europe
point = rlp.random_points('Europe')
print(point)



# Setting up Epochs
epc0 = bh.Epoch(2024, 5, 20, 0, 0, 0) # This is the epoch of the orbital elements
epc10 = epc0 + 86400

gs1 = bdm.Station(
            **{
                "properties": {
                    "constraints": bdm.AccessConstraints(elevation_min=0),
                    "name": "change_this",
                },
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [0, -90] # ASK IF THIS IS RIGHT 
                },
            }
        )

def cost_func_gap( gs ,satellites = satellites[0:3], epc_start = epc0, epc_end = epc10, plot = False):

    
    _, _, gaps_seconds = compute_all_gaps_contacts(satellites, [gs],epc_start, epc_end, plot)

    return 3*np.mean(gaps_seconds) + np.std(gaps_seconds)

lats = []
longs = []
gaps = []
for i in range(50):
    gs = rlp.random_points('Europe')[0]
    lon = gs[0]
    lat = gs[1]
    gs = return_bdm_gs(gs[0], gs[1])
    lats.append(lat)
    longs.append(lon)
    print(lat,lon)
    gap = cost_func_gap( gs ,satellites[0:3], epc0, epc10, False)
    gaps.append(gap)








# Plotting Imports
import shapely
import cartopy.crs as ccrs
import cartopy.geodesic
# Create the figure
fig = plt.figure(figsize=(10,5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.stock_img()
c = 'b' # Set the plot color

# Plot Groundstation Location
# ax.plot(lon, lat, color=c, marker='o', markersize=3, transform=ccrs.Geodetic())

# # Get a bunch of points in a circle space at the the right angle offset from the sub-satellite point to draw the view cone
# circle_points = cartopy.geodesic.Geodesic().circle(lon=lon, lat=lat, radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
# geom = shapely.geometry.Polygon(circle_points)
# ax.add_geometries((geom,), crs=ccrs.Geodetic(), facecolor=c, alpha=0.5, edgecolor='none', linewidth=0)



ax.set_yticks(np.arange(-90, 90, 30))
ax.set_xticks(np.arange(-180, 180, 30))
sc = ax.scatter(x=longs, y=lats, c=gaps,cmap = 'cool')
plt.colorbar(sc)
plt.grid()
plt.show()
