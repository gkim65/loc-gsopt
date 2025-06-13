import numpy as np
from scipy.optimize import differential_evolution
from common.objective_functions import cost_func_diffEvolution
from common.station_gen import return_bdm_gs
from common.utils import mp_compute_contact_times, xyz_to_latlon, latlon_to_xyz, contactExclusion

#  WandB
import wandb
import copy

# simplex selection
import random
from itertools import combinations
from shapely.geometry import Point, Polygon


def diffEvolution(cfg,land_data,epc_start,epc_end,satellites):

        # Setup args for minimize function
        gs_list = []
        gs_list_plot = []
        gs_contacts_og = []
        sat_list = satellites
        land_geometries = land_data['geometry']
        verbose = cfg.debug.verbose
        count = 1

        if cfg.debug.verbose:
            print("STARTING TO PERFORM MINIMIZATION ON ITERATION: "+str(count))

        # # Perform the optimization using Nelder-Mead
        result = differential_evolution(cost_func_diffEvolution, 
                        bounds = [(-180.0, 180.0), (-90.0, 90.0)] * cfg.problem.gs_num,
                        args = (sat_list, epc_start, epc_end, land_geometries, cfg, count, verbose, False), 
                        strategy='best1bin',)

                        
        if cfg.debug.verbose:
                print("Final GS list: "+str(result.x))

        print(result)
        # if cfg.debug.wandb:
        #         contacts, _ = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, False)
        #         _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
        #         wandb.summary["gs_list"+str(iterate)] = gs_list_plot 
        #         wandb.summary["contact_num"+str(iterate)] = len(contacts_exclusion_secs) 
        #         wandb.summary["seconds"+str(iterate)] = np.sum(contacts_exclusion_secs)
        #         wandb.summary["data_downlink"+str(iterate)] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate
        gs_list_plot =  [[lon, lat] for lon, lat in zip(result.x[::2], result.x[1::2])]
        gs_list = [return_bdm_gs(coord[0], coord[1]) for coord in gs_list_plot] 

                               
                
        return gs_list, gs_list_plot 

