import numpy as np
from scipy.optimize import minimize
from common.objective_functions import cost_func
import brahe as bh
from common.utils import compute_contact_times, xyz_to_latlon, latlon_to_xyz, contactExclusion

#  WandB
import wandb
import copy

# simplex selection
import random
from itertools import combinations
from shapely.geometry import Point, Polygon


def haversine_numba(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi  = np.deg2rad(lat2 - lat1)
    dlamb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlamb/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def farthest_four(points):
    """Greedy k‑centre: select 4 widely‑spaced points from `points` (N×2)."""
    N = len(points)
    chosen = [np.random.randint(N)]          # 1st point random
    dists  = np.full(N, np.inf)

    for _ in range(3):                      # need total of 4
        # update min distance to current set
        for i in range(N):
            for j in chosen[-1:]:
                d = haversine_numba(points[i,0], points[i,1],
                                     points[j,0], points[j,1])
                if d < dists[i]:
                    dists[i] = d
        chosen.append(int(np.argmax(dists))) # pick farthest

    return points[chosen]

# --- driver -------------------------------------------------
def simplex_select(n_samples=120):
    lats = np.degrees(np.arcsin(np.random.uniform(-1, 1, n_samples)))
    lons = np.random.uniform(-180, 180, n_samples)
    pts  = np.column_stack((lats, lons))
    return farthest_four(pts)




def nelder_mead_scipy_ccgs(cfg,land_data,epc_start,epc_end,satellites,eval_counter,city_data):

        # Setup args for minimize function
        if cfg.constraints.extra_gs:
                gs_list = [bh.PointLocation(cfg.constraints.extra_gs_lon,cfg.constraints.extra_gs_lat)]
                gs_list_plot = [[cfg.constraints.extra_gs_lon,cfg.constraints.extra_gs_lat]]
                gs_offset = 1

        else:
                gs_list = []
                gs_list_plot = []
                gs_offset = 0
        gs_contacts_og = []
        sat_list = satellites
        land_geometries = land_data['geometry']
        verbose = cfg.debug.verbose

        if cfg.problem.gs_num == 1 and cfg.constraints.extra_gs:
                return gs_list, gs_list_plot

        for iterate in range(cfg.experiments.ccgs):

                # for every ground station
                for i in range(gs_offset, cfg.problem.gs_num):

                # for i in range(cfg.problem.gs_num):
                

                        # Initial guess overridden by initial simplex
                        initial_guess = np.array([-0.41244896,  0.6765351 ,  0.61007058])

                        # initial_simplex = simplex_select(gs_list_plot,cfg.experiments.simplexExclude)
                        initial_simplex = simplex_select(n_samples=200)
                        if cfg.debug.wandb:
                                wandb.log({"initial simplex"+str(i+1): initial_simplex})
                                
                        # conversion of simplex to unit sphere
                        simplex = []
                        for p in initial_simplex:
                                simplex.append(latlon_to_xyz(p[1], p[0]))

                        if cfg.debug.verbose:
                                print("STARTING TO PERFORM MINIMIZATION ON GS: "+str(i+1))
                                print("lat-long simplex: ", initial_simplex)
                                print("unit circle converted simplex: ", simplex)

                        # Make faster computation
                        if iterate > 0:
                                # Get all ground stations except the i-th one
                                gs_list_others = [gs for idx, gs in enumerate(gs_list) if idx != i]
                                # try to minimize number of contacts to compute:
                                contacts_og, contacts_sec = compute_contact_times(satellites, gs_list_others ,epc_start, epc_end)
                                gs_contacts_og = contacts_sec

                        # # Perform the optimization using Nelder-Mead
                        result = minimize(cost_func, 
                                        initial_guess, 
                                        args = (gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og,eval_counter,city_data, verbose, False), 
                                        method='Nelder-Mead',
                                        options={'disp': True,
                                                'maxiter': 100,
                                                'xtol': 1,     # x tolerance
                                                'ftol': 1,     # function tolerance
                                                'initial_simplex': np.array(simplex)})
                                                #,
                                                # 'maxiter': 3})

                        
                        if cfg.debug.verbose:
                                print("GS FOUND, Location: "+str(result.x))

                        # conversion of unit circle coordinates back to lon,lat
                        if iterate == 0:
                                coord = xyz_to_latlon(result.x)
                                gs_list.append(bh.PointLocation(coord[1], coord[0]))
                                gs_list_plot.append([coord[1], coord[0]])

                                # try to minimize number of contacts to compute:
                                contacts_og, contacts_sec = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                                if i < cfg.problem.gs_num-1: # prevent double counting
                                        gs_contacts_og = contacts_sec
                        else:
                                coord = xyz_to_latlon(result.x)
                                gs_list_new = gs_list.copy() #copy.deepcopy(gs_list)
                                gs_list_new[i] = bh.PointLocation(coord[1], coord[0])

                                # Check if prev is better than current
                                contacts_prev, contacts_sec_prev = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                                # _, contacts_exclusion_secs_prev = contactExclusion(contacts_prev,cfg)
                                contacts_new, contacts_sec_new = compute_contact_times(satellites, gs_list_new ,epc_start, epc_end)
                                # _, contacts_exclusion_secs_new = contactExclusion(contacts_new,cfg)

                                if np.sum(contacts_sec_prev) < np.sum(contacts_sec_new): # or np.sum(contacts_exclusion_secs_prev) < np.sum(contacts_exclusion_secs_new):
                                        print(gs_list)
                                        print(gs_list_plot)
                                        gs_list[i] = bh.PointLocation(coord[1], coord[0])
                                        gs_list_plot[i] = [coord[1], coord[0]]

                

                if cfg.debug.wandb:
                        contacts, _ = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                        _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                        wandb.summary["gs_list"+str(iterate)] = gs_list_plot 
                        wandb.summary["contact_num"+str(iterate)] = len(contacts_exclusion_secs) 
                        wandb.summary["seconds"+str(iterate)] = np.sum(contacts_exclusion_secs)
                        wandb.summary["data_downlink"+str(iterate)] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate
                        wandb.summary["eval_counter"+str(iterate)] = eval_counter.score_count

                               
                
        return gs_list, gs_list_plot 



def powell_scipy(cfg,land_data,epc_start,epc_end,satellites,eval_counter,city_data):

        # Setup args for minimize function
        gs_list = []
        gs_list_plot = []
        gs_contacts_og = []
        sat_list = satellites
        land_geometries = land_data['geometry']
        verbose = cfg.debug.verbose

        for iterate in range(cfg.experiments.ccgs):
                # for every ground station
                for i in range(cfg.problem.gs_num):
                
                        # Latitude: Uniform sampling between -90 and 90 degrees
                        lat = random.uniform(-90, 90)
                        
                        # Longitude: Uniform sampling between -180 and 180 degrees
                        lon = random.uniform(-180, 180)

                        # Initial guess overridden by initial simplex
                        initial_guess = np.array([lon,lat])

                        if cfg.debug.verbose:
                                print("STARTING TO PERFORM MINIMIZATION ON GS: "+str(i+1))

                        if iterate > 0:
                                gs_list_others = [gs for idx, gs in enumerate(gs_list) if idx != i]
                                contacts_og, contacts_sec = compute_contact_times(
                                        satellites, gs_list_others, epc_start, epc_end
                                )
                                gs_contacts_og = contacts_sec

                        result = minimize(
                                        cost_func,
                                        initial_guess,
                                        args=(gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og,eval_counter,city_data, verbose, False), 
                                        method='Powell',
                                        bounds=[(-180, 180), (-90, 90)],
                                        options={
                                                'disp': True,
                                                'xtol': 1e-3,     # x tolerance
                                                'ftol': 1e-1,     # function tolerance
                                                'maxiter': 100
                                        }
                                        )
                        
                        
                        if cfg.debug.verbose:
                                print("GS FOUND, Location: "+str(result.x))

                        if iterate == 0:
                                gs_list.append(bh.PointLocation(result.x[0], result.x[1]))
                                gs_list_plot.append([result.x[0], result.x[1]])

                        else:
                                coord = result.x
                                gs_list_new = gs_list.copy()
                                gs_list_new[i] = bh.PointLocation(coord[0], coord[1])

                                contacts_prev, contacts_sec_prev = compute_contact_times(
                                satellites, gs_list, epc_start, epc_end
                                )
                                contacts_new, contacts_sec_new = compute_contact_times(
                                satellites, gs_list_new, epc_start, epc_end
                                )

                                if np.sum(contacts_sec_prev) < np.sum(contacts_sec_new):
                                        gs_list[i] = bh.PointLocation(coord[0], coord[1])
                                        gs_list_plot[i] = [coord[0], coord[1]]
                        # try to minimize number of contacts to compute:
                        contacts_og, contacts_sec = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                        gs_contacts_og = contacts_sec

                if cfg.debug.wandb:
                        contacts, _ = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                        _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                        wandb.summary["gs_list"+str(iterate)] = gs_list_plot 
                        wandb.summary["contact_num"+str(iterate)] = len(contacts_exclusion_secs) 
                        wandb.summary["seconds"+str(iterate)] = np.sum(contacts_exclusion_secs)
                        wandb.summary["data_downlink"+str(iterate)] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate
                        wandb.summary["eval_counter"+str(iterate)] = eval_counter.score_count
                
        return gs_list, gs_list_plot 


##### TODO EXTENSION: Powell / other minimize functions?
                # result = minimize(
                #                 cost_func,
                #                 initial_guess,
                #                 args=(gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og, verbose, plot), 
                #                 method='Powell',
                #                 bounds=[(-180, 180), (-90, 90)],
                #                 options={
                #                         'disp': True,
                #                         'xtol': 1e-3,     # x tolerance
                #                         'ftol': 1e-6,     # function tolerance
                #                         'maxiter': 100
                #                 }
                #                 )