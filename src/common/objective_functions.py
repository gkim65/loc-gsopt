from shapely.geometry import Point
from geopy.distance import geodesic

from common.station_gen import return_bdm_gs
from common.utils import compute_all_gaps_contacts

import numpy as np
from global_land_mask import globe

############################## Helper Functions ################################

def calculate_distance_to_land(point, land_geometries):
    # Initialize minimum distance and closest land point
    min_distance = float('inf')
    closest_land_point = None

    for land_poly in land_geometries:
        # Check if the point is on land
        if land_poly.contains(point):
            return 0, point  # The point is on land
        
        # Handle MultiPolygon or Polygon
        if land_poly.geom_type == "MultiPolygon":
            for poly in land_poly.geoms:  # Iterate over individual polygons
                nearest_point = poly.exterior.interpolate(
                    poly.exterior.project(point)
                )
                dist = geodesic((point.y, point.x), (nearest_point.y, nearest_point.x)).meters
                if dist < min_distance:
                    min_distance = dist
                    closest_land_point = nearest_point
        elif land_poly.geom_type == "Polygon":
            nearest_point = land_poly.exterior.interpolate(
                land_poly.exterior.project(point)
            )
            dist = geodesic((point.y, point.x), (nearest_point.y, nearest_point.x)).meters
            if dist < min_distance:
                min_distance = dist
                closest_land_point = nearest_point

    return min_distance, closest_land_point


# https://web.stanford.edu/group/sisl/k12/optimization/MO-unit5-pdfs/5.6penaltyfunctions.pdf
# For Quadratic Penalty/Loss function
def penalty(new_gs,land_geometries):
    on_water = False
    on_water = globe.is_ocean(new_gs[1], new_gs[0])
    if on_water:
        # Calculate distance to the nearest lan
        distance, closest_land_point = calculate_distance_to_land(Point(new_gs[0], new_gs[1]), land_geometries)
        return distance
    return 0

############################## Cost Functions ################################

# TODO: delete plot argument
def cost_func_gap(new_gs, gs_list, global_list_of_simplexes, satellites, epc_start, epc_end, land_geometries, plot = False):    
    
    gs_list = [return_bdm_gs(new_gs[0], new_gs[1])] # FIX FOR LATER! TODO
    global_list_of_simplexes.append([new_gs[0], new_gs[1]]) # CHECK IF IT WORKS TODO

    _, _, gaps_seconds = compute_all_gaps_contacts(satellites, gs_list ,epc_start, epc_end, plot)

    value = np.mean(gaps_seconds) + (penalty(new_gs,land_geometries)/1000)**2 # Put penalty/distance from land in kms

    return value