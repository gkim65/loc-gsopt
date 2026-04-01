import wandb
# Plotting Imports
import shapely
import cartopy.crs as ccrs
import cartopy.geodesic
import matplotlib.pyplot as plt
from common.utils import compute_earth_interior_angle
import brahe as bh
import numpy as np
from matplotlib.lines import Line2D

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

# Plotting Imports
def main():
    constellation = "CAPELLA Space"
    # constellation = "ICEYE"
    gs_KSAT, gs_list_teleports, gs_SCORE, gs_infra, gs_lat = KSAT_TeleportLists(constellation)

    alt = 570 # Altitude in km
    elevation_min = 10.0 # Keep to 10, but try plotting later 10 - 5
    lam = compute_earth_interior_angle(ele=elevation_min, alt=alt)

    fig = plt.figure(figsize =(11,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()

    # --- Teleports (purple triangles) ---
    for point in gs_list_teleports:
        circle_points = cartopy.geodesic.Geodesic().circle(lon=point[0], lat=point[1], radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
        geom = shapely.geometry.Polygon(circle_points)
        ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.2, edgecolor='none', linewidth=0, facecolor="purple")
    ax.scatter(*zip(*gs_list_teleports), transform=ccrs.PlateCarree(), color="purple", marker="^")

    # --- SCORE (blue circles) ---
    for point in gs_SCORE:
        circle_points = cartopy.geodesic.Geodesic().circle(lon=point[0], lat=point[1], radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
        geom = shapely.geometry.Polygon(circle_points)
        ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.2, edgecolor='none', linewidth=0, facecolor="blue")
    ax.scatter(*zip(*gs_SCORE), transform=ccrs.PlateCarree(), color="blue")

    # --- KSAT (red squares) ---
    for point in gs_KSAT:
        circle_points = cartopy.geodesic.Geodesic().circle(lon=point[0], lat=point[1], radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
        geom = shapely.geometry.Polygon(circle_points)
        ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.2, edgecolor='none', linewidth=0, facecolor="red")
    ax.scatter(*zip(*gs_KSAT), transform=ccrs.PlateCarree(), color="red", marker="s")

    # --- Infrastructure (orange diamonds) ---
    for point in gs_infra:
        circle_points = cartopy.geodesic.Geodesic().circle(lon=point[0], lat=point[1], radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
        geom = shapely.geometry.Polygon(circle_points)
        ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.2, edgecolor='none', linewidth=0, facecolor="orange")
    ax.scatter(*zip(*gs_infra), transform=ccrs.PlateCarree(), color="orange", marker="D")

    # --- Latitude-optimized (green pentagons) ---
    for point in gs_lat:
        circle_points = cartopy.geodesic.Geodesic().circle(lon=point[0], lat=point[1], radius=lam*bh.R_EARTH, n_samples=100, endpoint=False)
        geom = shapely.geometry.Polygon(circle_points)
        ax.add_geometries((geom,), crs=ccrs.Geodetic(), alpha=0.2, edgecolor='none', linewidth=0, facecolor="gold")
    ax.scatter(*zip(*gs_lat), transform=ccrs.PlateCarree(), color="gold", marker="p")

    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xticks(np.arange(-180, 181, 30))
    plt.setp(ax.get_xticklabels(), rotation=50, ha='center', va='top')
    ax.tick_params(labelsize=23)

    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='SCORE',         markerfacecolor='blue',   markersize=10),
        Line2D([0], [0], marker='s', color='none', label='KSAT',          markerfacecolor='red',    markersize=10),
        Line2D([0], [0], marker='^', color='none', label='Teleports',     markerfacecolor='purple', markersize=10),
        Line2D([0], [0], marker='D', color='none', label='Infrastructure',markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='p', color='none', label='Latitude',      markerfacecolor='gold',  markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=14)

    plt.grid()

    if constellation == "CAPELLA Space":
        plt.savefig("figures_final/capella_comparison.pdf")
    if constellation == "ICEYE":
        plt.savefig("figures_final/iceye_comparison.pdf")


def KSAT_TeleportLists(constellation):
    if constellation == "CAPELLA Space":
        gs_SCORE = [
        [107.86566608986948, 41.03742259653801],
        [-105.97680974315811, 41.52407666708533],
        [148.337001, -40.388591],
        [-177.380603, 28.211468 ],
        [15.09540884447382, 41.64324004063826],
        [51.727753, -46.282698, ],
        [1.92019974253284, -89.93573497119367],
        [-62.77158633949658, -40.30308683262527],
        [-59.882177, 43.930450, ],
        [60.96547761528194, 40.84103881603662]
        ]

        gs_KSAT = [
        [168.38, -46.53],      # Awarua
        [-25.13, 36.99],       # Azores
        [-118.15, 33.82],      # Long Beach
        [-84.26, 32.95],       # Thomaston
        [27.71, -25.89],       # Hartebeesthoek
        [-156.45, 20.82],      # Hawaii
        [143.45, 42.6],        # Hokkaido
        [115.34, -29.01],      # Mingenew
        [-70.87, -52.94],      # Punta Arenas
        [22.69, 38.82]         # Thermopylae
        ]
        gs_list_teleports= [
        [18.71868, -34.02762],
        [-157.8664848, 21.31494],
        [141.2583617, 43.1732192],
        [168.3811515, -46.5288827],
        [115.941223, -31.880318],
        [-111.951599, 40.783137],
        [-52.7755931, 47.5609626],
        [-3.792241, 40.39759],
        [44.904557, 41.6955548],
        [-58.312699, -34.688141]
        ]
        gs_lat = [[-8.137314714308896, 41.030701242265096], [68.77371747901836, -48.49793763012599], [-176.95105931701485, 51.48588116004088], [173.01970837708168, -40.7668828486862], [109.720896205166, 41.01357662571634], [-109.68647325423794, 40.49748622815677], [136.68166381739834, -36.168275379201674], [-62.79417668832446, -40.32302229454601], [20.00079630405991, -34.909998188938204], [60.96513521691464, 40.841975620542506]]
        gs_infra = [[-174.66660910628372, -18.83340553524406], [-171.29253398878112, 52.34575328143677], [140.89171696899243, 41.99176922144044], [-60.877959481069778, -40.392342215157505], [136.68180623502363, -36.167955315652065], [-124.38771358426798, 40.25321439491545], [23.53101150937204, -34.77408582928908], [-81.37532302818501, 40.97784006586425], [-4.408178428068733, 51.00000880745846], [44.51610742464326, 41.44940009938876]]
    else:
        gs_KSAT = [
        [148.49, 70.26],  # Alaska, USA
        [168.38, -46.53],  # New Zealand
        [143.45, 42.6],  # Japan
        [-133.72, 68.36],  # Northwest Territories, Canada
        [-8.62, 71.0],  # Jan Mayen Island, Norway
        [-51.72, 64.18],  # Greenland
        [15.63, 78.25],  # Svalbard, Norway
        [-67.52, -55.05],  # Tierra del Fuego, Argentina
        [2.53, -72.01],  # Troll Station, Antarctica
        [31.11, 70.37]  # Vardo, Norway
        ]
        gs_list_teleports =  [
                [30.504861111111,50.46615],
                [168.3811515,-46.5288827],
                [93.5143972,56.24975],
                [15.390987,78.229206],
                [-67.115683,-54.510862],
                [2.534985,-72.011643],
                [-21.687227,64.143157],
                [-133.549226,68.3195115],
                [-148.9412693,70.3178746],
                [-80.3336725,41.4337083]
                ]
        gs_SCORE = [[-127.31584941014852,57.77654962067853],[66.5179763779409,-89.99605283453106],[-32.31741688781365,85.3501219671681],[-74.13385227216493,59.81151712271469],[158.9609823182852,-55.140339937067594],[102.63914516788462,66.60441695196317],[-8.13271233106455,58.186186375006315],[175.4796711118354,67.32552950216812],[46.22404223420632,57.47260533749195],[-67.62880162677172,-56.20996466448185]]
        gs_infra = [[2.534985, -72.011643], [166.53704277835445, -48.265647537825345], [12.688803950991463, 79.06671133793076], [28.12849494563114, -33.16317206280653], [-13.822091235321562, 58.05802238796434], [115.29480009495336, -33.65833284018386], [-133.5750303955949, 69.02403072502598], [137.11319900070305, 37.609817995195776], [-61.62886485943242, 53.08114537384188], [-67.19594921851454, -56.25000328401075]]
        gs_lat = [[2.534985, -72.011643], [11.688803950991463, 79.06671133793076], [-113.6170936611067, 70.74391252788841], [166.55990114964908, -48.261683932421526], [149.9973743386689, 72.26820523711807], [82.45625521011117, 61.40008563582113], [-67.19594693902962, -56.249954287248265], [-7.351465961785391, 52.96382401211473], [70.11108965630885, -49.96939671017328], [-164.28150681040967, 53.45206155148134]]

    return gs_KSAT, gs_list_teleports, gs_SCORE, gs_infra, gs_lat
    
    

if __name__ == "__main__":
    main()