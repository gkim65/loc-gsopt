from src.sat_gen import satellites_from_constellation
from src.station_gen import gs_json, rand_gs_on_land,return_bdm_gs
from src.utils import load_earth_data, compute_all_gaps_contacts, compute_earth_interior_angle
from src.optimizations.nelder_mead import nelder_mead

import numpy as np

# Brahe Imports
import brahe as bh
import brahe.data_models as bdm
import brahe.access.access as ba

# random points function
import random_land_points as rlp

# Plotting Imports
import shapely
import cartopy.crs as ccrs
import cartopy.geodesic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


############################### STEP 3: Scenario Generation ###############################

# Setting up Epochs
epc0 = bh.Epoch(2024, 5, 20, 0, 0, 0) # This is the epoch of the orbital elements
epc10 = epc0 + 86400 # TODO: make this longer time period

points = []
for i in range(3):
    point = rlp.random_points('Antarctica')[0] # Get a random point on land in Antartica
    points.append((point[0],point[1]))


# Cost function, we should be able tochange this if needed?
def cost_func_gap(new_gs, gs_list = [], satellites=satellites[0:3], epc_start = epc0, epc_end = epc10, plot = False):    
    gs_list.append(return_bdm_gs(new_gs[0], new_gs[1]))
    _, _, gaps_seconds = compute_all_gaps_contacts(satellites, gs_list ,epc_start, epc_end, plot)
    return np.mean(gaps_seconds)

print(points)
plot_points = nelder_mead(points, cost_func_gap, 10)
print(plot_points)


############################### STEP 4: Plotting ###############################

# Now we can plot what a satellite can see from a given altitude
# Let's say we want to see what a satellite at 525 km can see if all observers
# are looking at it with at least 20 degrees elevation angle. This is an appropriate
# Value for a communications (Starlink/Kuiper) user terminal. If you wanted to get
# the maximum possible coverage limited by the Earth's curvature, you would use 0 degrees.

alt = 570 # Altitude in km
elevation_min = 20.0
lam = compute_earth_interior_angle(ele=elevation_min, alt=alt)


def animate(i):
    fig.clear()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()
    points = plot_points[i]
    print(i)
    print(points)

    ax.set_title("Animation Counter Optimization: "+ str(i))
    ax.scatter(*zip(*points))
    for point in points:
        # Get a bunch of points in a circle space at the the right angle offset from the sub-satellite point to draw the view cone
        circle_points = cartopy.geodesic.Geodesic().circle(lon=point[0], lat=point[1], radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
        geom = shapely.geometry.Polygon(circle_points)
        ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.5, edgecolor='none', linewidth=0)

    ax.set_yticks(np.arange(-90, 90, 30))
    ax.set_xticks(np.arange(-180, 180, 30))
    # sc = ax.scatter(x=longs, y=lats, c=gaps,cmap = 'cool')
    # plt.colorbar(sc)
    plt.grid()
    # plt.show()


print(len(plot_points))

fig = plt.figure()
ani = FuncAnimation(fig, animate, frames = range(len(plot_points)),repeat = False,interval =500)
# plt.ylim(-1000,1000)
# plt.xlim(-1000,1000)
# plt.show()
ani.save(filename="example.gif", writer="ffmpeg")
