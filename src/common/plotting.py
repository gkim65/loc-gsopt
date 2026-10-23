from common.utils import compute_earth_interior_angle
import numpy as np
import brahe as bh


# Plotting Imports
import shapely
import cartopy.crs as ccrs
import cartopy.geodesic
import matplotlib.pyplot as plt
from common.utils import compute_gaps_per_sat
import warnings
# Suppress Cartopy's CRS approximation warnings
warnings.filterwarnings("ignore", category=UserWarning)

############################## Helper Functions ################################

alt = 570 # Altitude in km
elevation_min = 10.0 # Keep to 10, but try plotting later 10 - 5
lam = compute_earth_interior_angle(ele=elevation_min, alt=alt)


def animate(i,fig,plot_points,title = "Animation Counter Optimization:",gs_list=[],ind=0):
    fig.clear()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()
    if type(i) == int:
        points = plot_points[i]
    else:
        points = plot_points

    ax.set_title(title+ str(i))
    ax.scatter(*zip(*points), transform=ccrs.PlateCarree())
    for point in points:
        # Get a bunch of points in a circle space at the the right angle offset from the sub-satellite point to draw the view cone
        circle_points = cartopy.geodesic.Geodesic().circle(lon=point[0], lat=point[1], radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
        geom = shapely.geometry.Polygon(circle_points)
        ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.3, edgecolor='none', linewidth=0)

    if gs_list:
        ax.scatter(*zip(*gs_list[0:ind+1]), transform=ccrs.PlateCarree(),color = "red")
        for point in gs_list[0:ind+1]:
            # Get a bunch of points in a circle space at the the right angle offset from the sub-satellite point to draw the view cone
            circle_points = cartopy.geodesic.Geodesic().circle(lon=point[0], lat=point[1], radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
            geom = shapely.geometry.Polygon(circle_points)
            ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.3, edgecolor='none', facecolor='red', linewidth=0)

    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xticks(np.arange(-180, 181, 30))
    plt.grid()


############################## Main Plotting Functions ################################

def plot_img(gs_list,name="example.png"):
    fig = plt.figure()
    print(gs_list)
    animate("", fig, gs_list,"Full Ground Station Selection")

    plt.savefig(name)
    return fig

def plot_gap_times(satellites, ground_stations, epc_start, epc_end, plot, title="gap_times_chart.png"):
    if plot:
        _,_,gap_secs = compute_gaps_per_sat(satellites,ground_stations,epc_start,epc_end, plot,title)
        plt.savefig(title)
        return title
    return fig
