from common.utils import compute_earth_interior_angle
import numpy as np
import brahe as bh


# Plotting Imports
import shapely
import cartopy.crs as ccrs
import cartopy.geodesic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

############################## Helper Functions ################################

def sliding_window(data, window_size=3):
    """
    Creates a list of lists by applying a sliding window over the input list.

    Args:
        data (list): The input list to process.
        window_size (int): The size of the window (default is 3).

    Returns:
        list: A list of sublists, each containing `window_size` elements.
    """
    if window_size > len(data):
        raise ValueError("Window size cannot be larger than the data length.")

    result = [data[i:i + window_size] for i in range(len(data) - window_size + 1)]
    return result

alt = 570 # Altitude in km
elevation_min = 20.0
lam = compute_earth_interior_angle(ele=elevation_min, alt=alt)


def animate(i,fig,plot_points):
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
        ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.3, edgecolor='none', linewidth=0)

    ax.set_yticks(np.arange(-90, 90, 30))
    ax.set_xticks(np.arange(-180, 180, 30))
    # sc = ax.scatter(x=longs, y=lats, c=gaps,cmap = 'cool')
    # plt.colorbar(sc)
    plt.grid()
    # plt.show()


############################## Main Plotting Functions ################################

def plot_gif(global_list_of_simplexes):
    plot_points = sliding_window(global_list_of_simplexes)
    fig = plt.figure()
    ani = FuncAnimation(fig, animate, fargs=(fig,plot_points),frames = range(len(plot_points)),repeat = False,interval =500)

    ani.save(filename="example.gif", writer="ffmpeg")
