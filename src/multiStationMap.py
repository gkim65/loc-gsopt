import wandb
# Plotting Imports
import shapely
import cartopy.crs as ccrs
import cartopy.geodesic
import matplotlib.pyplot as plt
from common.utils import compute_earth_interior_angle
import brahe as bh
import numpy as np

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size" : 24, 
    "font.serif": ["Computer Modern Roman"],  # optional: you can specify others like Times
    "axes.unicode_minus": False  # optional: fix minus signs in LaTeX
})
# api = wandb.Api()
# project_name = f"loc_gsopt/heatmapfree_nelder_ccgs_data_downlink_4_4=CAPELLA=3000000"
# runs = api.runs(project_name)

runs = [
        # Set 1
        [[15.65, 78.23],      # Svalbard, Norway (78°N)
                [2.53, -72.01],      # Troll Station, Antarctica (72°S)
                [-133.72, 68.36],    # Inuvik, Canada (68°N)
                [-57.85, -51.68]],    # Stanley, Falkland Islands (52°S)

        # Set 2
        [[-26.51, 64.14],     # Reykjavik, Iceland (64°N)
                [2.53, -72.01],      # Troll Station, Antarctica (72°S)
                [-148.49, 70.26],    # Prudhoe Bay, Alaska (70°N)
                [168.38, -46.53]],    # Awarua, New Zealand (47°S)

        # Set 3
        [[25.75, 71.17],      # Vardo, Norway (71°N)
                [2.53, -72.01],      # Troll Station, Antarctica (72°S)
                [-51.72, 64.18],     # Nuuk, Greenland (64°N)
                [-70.87, -52.94]],
        [[31.11, 70.37],      # Vardo, Norway (70°N)
           [2.53, -72.01],      # Troll Station, Antarctica (72°S)
           [-68.70, 76.53],     # Baffin Island, Canada (77°N)
           [51.73, -46.28]],
        [[-68.70, 76.53],     # Baffin Island, Canada (77°N)
        [2.53, -72.01],      # Troll Station, Antarctica (72°S)
        [25.75, 71.17],      # Vardo, Norway (71°N)
        [-57.85, -51.68]],   # Stanley, Falkland Islands (52°S)
]
for (i,run) in enumerate(runs):
    gs_list = run#runs[run_num]
    print("HERHEHEHHEHEHEHHEEHHEHHE")
    # print(run.summary.data_downlink*52/1000/1000/1000/1000/1000/8)
    # print(gs_list)
    alt = 570 # Altitude in km
    elevation_min = 10.0 # Keep to 10, but try plotting later 10 - 5
    lam = compute_earth_interior_angle(ele=elevation_min, alt=alt)


    fig = plt.figure(figsize =(11,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()

    # ax.set_title("KSAT Ground Station Network")
    for point in gs_list:
        # Get a bunch of points in a circle space at the the right angle offset from the sub-satellite point to draw the view cone
        circle_points = cartopy.geodesic.Geodesic().circle(lon=point[0], lat=point[1], radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
        geom = shapely.geometry.Polygon(circle_points)
        ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.2, edgecolor='none', linewidth=0, facecolor = "darkblue")

    ax.scatter(*zip(*gs_list), transform=ccrs.PlateCarree(), color = "darkblue",s = 80)
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xticks(np.arange(-180, 181, 30))
    plt.setp(ax.get_xticklabels(), rotation=50, ha='center', va='top')
    ax.tick_params(labelsize=30)
    plt.grid()
    plt.tight_layout()


    plt.savefig(f"figures_final/multiStation{i}.pdf")
