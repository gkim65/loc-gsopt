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
# Suppress Cartopy's CRS approximation warnings.
# NOTE: this silences every UserWarning in the process once this module is imported.
warnings.filterwarnings("ignore", category=UserWarning)

############################## Helper Functions ################################

# Coverage circles on the station maps: the area that sees a satellite at `alt` above `elevation_min`
alt = 570 # Altitude in km
elevation_min = 10.0 # deg
lam = compute_earth_interior_angle(ele=elevation_min, alt=alt) # rad


def animate(i,fig,plot_points,title = "Animation Counter Optimization:",gs_list=[],ind=0):
    """
        animate(i, fig, plot_points, title="Animation Counter Optimization:", gs_list=[], ind=0) -> None

    Draw stations and their coverage circles on a world map in `fig`.

      - `i` — frame index into `plot_points`, or a non-int to draw `plot_points` itself
      - `fig` — matplotlib figure to clear and draw into
      - `plot_points` — `[lon, lat]` pairs (deg), or a sequence of such lists when `i` is an int
      - `title` — title prefix; `i` is appended
      - `gs_list` — extra `[lon, lat]` pairs (deg) drawn in red, up to index `ind`
      - `ind` — last index of `gs_list` to draw
    """
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
    """
        plot_img(gs_list, name="example.png") -> matplotlib.figure.Figure

    Save a world map of the stations and their coverage circles.

      - `gs_list` — `[lon, lat]` pairs (deg)
      - `name` — output image path

    Returns the figure.
    """
    fig = plt.figure()
    print(gs_list)
    animate("", fig, gs_list,"Full Ground Station Selection")

    plt.savefig(name)
    return fig

def plot_gap_times(satellites, ground_stations, epc_start, epc_end, plot, title="gap_times_chart.png"):
    """
        plot_gap_times(satellites, ground_stations, epc_start, epc_end, plot, title="gap_times_chart.png") -> str

    Save a chart of contacts and coverage gaps per satellite.

      - `satellites`, `ground_stations` — spacecraft and stations
      - `epc_start`, `epc_end` — simulation window (`bh.Epoch`)
      - `plot` — draw the chart when True
      - `title` — output image path

    Returns `title`.

    NOTE: relies on `compute_gaps_per_sat`, which needs the pre-1.0 brahe API; with `plot=False`
    it raises `NameError`.
    """
    if plot:
        _,_,gap_secs = compute_gaps_per_sat(satellites,ground_stations,epc_start,epc_end, plot,title)
        plt.savefig(title)
        return title
    return fig
