from shapely.geometry import Point
from geopy.distance import geodesic
import brahe as bh

from common.utils import compute_gaps_per_sat, compute_contact_times, contactExclusion, xyz_to_latlon, distance_to_nearest_city_km

from itertools import chain, combinations

import numpy as np
from global_land_mask import globe

#  WandB
import wandb

# At the top of objective_functions.py
class EvalCounter:
    def __init__(self):
        self.score_count = 0      # counts SCORE/Nelder-Mead evals
        self.de_count = 0         # counts DE evals
############################## Helper Functions ################################
def calculate_distance_to_land(point, land_geometries):
    min_distance = float('inf')
    closest_land_point = None

    for land_poly in land_geometries:
        # Check if the point is on land
        if land_poly.contains(point):
            return 0, point
        
        # Handle MultiPolygon or Polygon
        if land_poly.geom_type == "MultiPolygon":
            for poly in land_poly.geoms:
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

def calculate_lat_penalty(lon,lat, lat_bot, lat_top):
    if lat < lat_bot:
        return geodesic((lat_bot, lon), (lat, lon)).meters
    elif lat > lat_top:
        return geodesic((lat_top, lon), (lat, lon)).meters
    return 0

# https://web.stanford.edu/group/sisl/k12/optimization/MO-unit5-pdfs/5.6penaltyfunctions.pdf
# For Quadratic Penalty/Loss function
# def penalty(new_gs,land_geometries,lat_bot, lat_top):
#     on_water = False
#     on_water = globe.is_ocean(new_gs[1], new_gs[0])
#     if on_water:
#         # Calculate distance to the nearest lan
#         distance, closest_land_point = calculate_distance_to_lat(Point(new_gs[0], new_gs[1]), lat_bot, lat_top, land_geometries)
#         #calculate_distance_to_land(Point(new_gs[0], new_gs[1]), land_geometries)
#         return distance
#     return 0
def penalty(new_gs, land_geometries, lat_bot, lat_top):
    lon = new_gs[0]
    lat = new_gs[1]

    # ALWAYS compute latitude penalty
    lat_penalty = calculate_lat_penalty(lon,lat, lat_bot, lat_top)

    on_water = globe.is_ocean(lat, new_gs[0])

    if on_water:
        distance, _ = calculate_distance_to_land(
            Point(new_gs[0], lat), land_geometries
        )
        return distance + lat_penalty

    return lat_penalty

    

# Additional penalty when we have close location in gs?
def penalty_gs_all(new_gs,current_gs_list, dist_penalty):
    penalty_sum = 0
    for current_gs in current_gs_list:
        dist = geodesic((new_gs[1], new_gs[0]), (current_gs.latitude(bh.AngleFormat.DEGREES), current_gs.longitude(bh.AngleFormat.DEGREES))).meters
        if dist < dist_penalty:
            penalty_sum += dist_penalty - dist
    return penalty_sum

# ── Penalty ────────────────────────────────────────────────────────────────
def penalty_infrastructure(lon, lat,city_data, weight=1.0):
    """Quadratic penalty for distance from nearest city (in km)."""
    if weight > 0:

        on_water = globe.is_ocean(lat, lon)
        land_geometries = list(city_data['geometry'])  # GeoSeries → list of Shapely Polygons
        if not on_water:
            distance, _ = calculate_distance_to_land(
                Point(lon, lat), land_geometries
            )
            return distance*weight

        return 0

    else:
        return 0


# ── DE version (takes full gs_list) ───────────────────────────────────────
def penalty_infrastructure_diffEvolution(gs_list,city_data, weight=1.0):
    total = 0.0
    for gs in gs_list:
        lon = gs.longitude(bh.AngleFormat.DEGREES)
        lat = gs.latitude(bh.AngleFormat.DEGREES)
        total += penalty_infrastructure(lon, lat, city_data, weight)
    return total

def penalty_water_diffEvolution(gs_list,land_geometries, lat_bot, lat_top):
    
    on_water = False
    total_penalty = 0


    for gs in gs_list:
        # ALWAYS compute latitude penalty
        lat_penalty = calculate_lat_penalty(gs.longitude(bh.AngleFormat.DEGREES), gs.latitude(bh.AngleFormat.DEGREES), lat_bot, lat_top)
        on_water = globe.is_ocean(gs.latitude(bh.AngleFormat.DEGREES), gs.longitude(bh.AngleFormat.DEGREES))
        if on_water:
            # Calculate distance to the nearest lan
            distance, closest_land_point = calculate_distance_to_land(Point((gs.longitude(bh.AngleFormat.DEGREES), gs.latitude(bh.AngleFormat.DEGREES))), land_geometries)
            total_penalty += distance
    return total_penalty


# Penalty for any pair of ground stations that are too close
def penalty_gs_all_diffEvolution(gs_list, dist_penalty):
    penalty_sum = 0
    # Iterate over all unique pairs of ground stations
    for gs1, gs2 in combinations(gs_list, 2):
        # Extract (lat, lon) from each
        coord1 = (gs1.latitude(bh.AngleFormat.DEGREES), gs1.longitude(bh.AngleFormat.DEGREES))
        coord2 = (gs2.latitude(bh.AngleFormat.DEGREES), gs2.longitude(bh.AngleFormat.DEGREES))
        
        dist = geodesic(coord1, coord2).meters
        if dist < dist_penalty:
            penalty_sum += dist_penalty - dist
    return penalty_sum


############################## Cost Functions ################################


def cost_func(x, gs_list, satellites, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og,eval_counter,city_data, verbose = False, plot = False):    

    if cfg.problem.method.startswith("nelder"):
        # Normalize input to unit vector
        x_unit = x / np.linalg.norm(x)
        
        # Convert to lat/lon
        lat_deg, lon_deg = xyz_to_latlon(x_unit)
        new_gs = [lon_deg,lat_deg]

    if cfg.problem.method == "powell":
        new_gs = [x[0],x[1]]

    # Make sure that all ground stations are set to only add onto the existing selected constellations
    temp_gs_list = gs_list.copy()
    if not gs_list:
        temp_gs_list = [bh.PointLocation(new_gs[0], new_gs[1])]
    else:
        temp_gs_list.append(bh.PointLocation(new_gs[0], new_gs[1]))

    # Computing specific objective
    if cfg.problem.objective == "gap_optimization":
        _, _, gaps_seconds = compute_gaps_per_sat(satellites, temp_gs_list ,epc_start, epc_end, plot)

        # TODO: put in additional stats per satellite etc for mean, for now just flatten everything 
        gaps_seconds_flattened = list(chain.from_iterable(gaps_seconds))
        mean_gap_time = np.mean(gaps_seconds_flattened)
        cost_func_val = mean_gap_time

    if cfg.problem.objective == "max_contacts":
        all_contacts, _ = compute_contact_times(satellites, [bh.PointLocation(new_gs[0], new_gs[1])] ,epc_start, epc_end)
        cost_func_val = 0 - len(all_contacts)*100 # TODO: do we just multiply by a diff num?
 
    if cfg.problem.objective == "data_downlink":
        all_contacts, contacts_sec = compute_contact_times(satellites, [bh.PointLocation(new_gs[0], new_gs[1])] ,epc_start, epc_end)
        cost_func_val = 0 - (np.sum(contacts_sec)+ np.sum(gs_contacts_og))

    penalty_water = (penalty(new_gs,land_geometries,cfg.constraints.latitude_bot,cfg.constraints.latitude_top)/1000)**2 # Put penalty/distance from land in 10 kms
    penalty_close_gs = (penalty_gs_all(new_gs,gs_list, cfg.constraints.dist_other_gs))**2 # additional penalty being close to gs, in ms
    penalty_infra = (penalty_infrastructure(
        new_gs[0], new_gs[1],
        city_data,
        cfg.constraints.infra_weight
    )/1000)**2
    value = cost_func_val + penalty_water + penalty_close_gs + penalty_infra
    # value = cost_func_val + penalty_water + penalty_close_gs
    eval_counter.score_count += 1

    if cfg.debug.wandb:
        wandb.log({
            "Obj_func_value": value,
            "penalty_water": penalty_water,
            "penalty_close_gs": penalty_close_gs,
            "penalty_infra": penalty_infra,          # ── new
            "log_of_simplexes_lon" + str(i+1): new_gs[0],
            "log_of_simplexes_lat" + str(i+1): new_gs[1]
        })
    if verbose:
        print(new_gs)
        print("Current optimization value: ", value)
        print("penalty_water: ", penalty_water)
        print("penalty_close_gs: ", penalty_close_gs)
        print("penalty_infra: ", penalty_infra)
        print(len(contacts_sec))
    
    return value



def cost_func_diffEvolution(x, satellites, epc_start, epc_end, land_geometries, cfg, count,eval_counter,city_data, verbose = False, plot = False):    

    # Make sure that all ground stations are set to only add onto the existing selected constellations
    gs_list_plot =  [[lon, lat] for lon, lat in zip(x[::2], x[1::2])]
    temp_gs_list =  [bh.PointLocation(lon, lat) for lon, lat in zip(x[::2], x[1::2])]

    # Computing specific objective
    if cfg.problem.objective == "gap_optimization":
        _, _, gaps_seconds = compute_gaps_per_sat(satellites, temp_gs_list ,epc_start, epc_end, plot)

        # TODO: put in additional stats per satellite etc for mean, for now just flatten everything 
        gaps_seconds_flattened = list(chain.from_iterable(gaps_seconds))
        mean_gap_time = np.mean(gaps_seconds_flattened)
        cost_func_val = mean_gap_time

    if cfg.problem.objective == "max_contacts":
        all_contacts, _ = compute_contact_times(satellites, temp_gs_list ,epc_start, epc_end)
        cost_func_val = 0 - len(all_contacts)*100 # TODO: do we just multiply by a diff num?
 
    if cfg.problem.objective == "data_downlink":
        all_contacts, contacts_sec = compute_contact_times(satellites, temp_gs_list ,epc_start, epc_end)
        cost_func_val = 0 - np.sum(contacts_sec)

    penalty_water = (penalty_water_diffEvolution(temp_gs_list,land_geometries,cfg.constraints.latitude_bot,cfg.constraints.latitude_top)/1000)**2 # Put penalty/distance from land in 10 kms
    penalty_close_gs = (penalty_gs_all_diffEvolution(temp_gs_list, cfg.constraints.dist_other_gs))**2 # additional penalty being close to gs, in ms
    penalty_infra = (penalty_infrastructure_diffEvolution(
        temp_gs_list,
        city_data,
        cfg.constraints.infra_weight
    )/1000)**2
    value = cost_func_val + penalty_water + penalty_close_gs + penalty_infra
    eval_counter.de_count += 1

    if cfg.debug.wandb:
        wandb.log({
            "Obj_func_value": value,
            "penalty_water": penalty_water,
            "penalty_close_gs": penalty_close_gs,
            "penalty_infra": penalty_infra,          # ── new
            "gs_list" + str(count): gs_list_plot
        })
        count += 1

    if verbose:
        print("Current optimization value: ", value)
        print("penalty_water: ", penalty_water)
        print("penalty_close_gs: ", penalty_close_gs)
        print("penalty_infra: ", penalty_infra)
        print(len(contacts_sec))
    
    return value

