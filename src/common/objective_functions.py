from shapely.geometry import Point
from geopy.distance import geodesic

from common.station_gen import return_bdm_gs
from common.utils import compute_gaps_per_sat, compute_contact_times

from itertools import chain
import numpy as np
from global_land_mask import globe

#  WandB
import wandb

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

# Additional penalty when we have close location in gs?
def penalty_gs_all(new_gs,current_gs_list, dist_penalty):
    penalty_sum = 0
    for current_gs in current_gs_list:
        dist = geodesic((new_gs[1], new_gs[0]), (current_gs.geometry.coordinates[1], current_gs.geometry.coordinates[0])).meters
        if dist < dist_penalty:
            penalty_sum += dist_penalty - dist
    return penalty_sum


############################## Cost Functions ################################

def cost_func(new_gs, gs_list, satellites, epc_start, epc_end, land_geometries, cfg, i, verbose = False, plot = False):    
    
    # Make sure that all ground stations are set to only add onto the existing selected constellations
    temp_gs_list = gs_list.copy()
    if not gs_list:
        temp_gs_list = [return_bdm_gs(new_gs[0], new_gs[1])]
    else:
        temp_gs_list.append(return_bdm_gs(new_gs[0], new_gs[1]))

    # Computing specific objective
    if cfg.problem.objective == "minimize_gap":
        _, _, gaps_seconds = compute_gaps_per_sat(satellites, temp_gs_list ,epc_start, epc_end, plot)

        # TODO: put in additional stats per satellite etc for mean, for now just flatten everything 
        gaps_seconds_flattened = list(chain.from_iterable(gaps_seconds))
        mean_gap_time = np.mean(gaps_seconds_flattened)
        cost_func_val = mean_gap_time

    if cfg.problem.objective == "maximize_contact_seconds":
        contact_times, contact_times_seconds = compute_contact_times(satellites, temp_gs_list ,epc_start, epc_end, plot)
        cost_func_val = 0 - (np.sum(contact_times_seconds))

    penalty_water = (penalty(new_gs,land_geometries)/10000)**2 # Put penalty/distance from land in 10 kms
    penalty_close_gs = (penalty_gs_all(new_gs,gs_list, cfg.constraints.dist_other_gs))**2 # additional penalty being close to gs, in ms
    value = cost_func_val + penalty_water + penalty_close_gs

    wandb.log({"Obj_func_value": value,
               "penalty_water": penalty_water,
                "penalty_close_gs": penalty_close_gs,
                 "log_of_simplexes_lon"+str(i): new_gs[0],
                 "log_of_simplexes_lat"+str(i):new_gs[1]})
    if verbose:
        print("Current optimization value: ", value)
        print("penalty_water: ", penalty_water)
        print("penalty_close_gs: ", penalty_close_gs)
    
    return value

