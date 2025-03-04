import numpy as np
from scipy.optimize import minimize
from common.objective_functions import cost_func
from common.station_gen import return_bdm_gs

#  WandB
import wandb
import copy


# TODO: add the cyclic coordinate descent part
def nelder_mead_scipy(cfg,land_data,epc_start,epc_end,satellites):
        

        # Setup args for minimize function
        gs_list = []
        gs_list_plot = []
        
        sat_list = satellites[0:cfg.problem.sat_num]
        land_geometries = land_data['geometry']
        verbose = cfg.debug.verbose
        plot = cfg.debug.plot

        for i in range(cfg.problem.gs_num):
        
                # Initial guess for the coordinates (x, y) 
                # Overridden currently by initial simplex
                initial_guess = np.array([-30, -60])

                # TODO: Find best initial simplex to start, may need to change based on continents
                # initial_simplex =  np.array([[-30, -60],[-100, -60],[-30, -90]])
                # initial_simplex =  np.array([[-180, -90],[180, 0],[-180, 90]])
                
                initial_simplex = np.array([
                        [-60, -85],  # Point 1 (near the Caribbean)
                        [151.2093, -33.8688],  # Point 2 (Sydney, Australia)
                        [-153.67891140271288, 55.17076207063536]  # Point 3 (near Alaska)
                        ])
                print(initial_simplex)

                
                print("STARTING TO PERFORM MINIMIZATION ON GS: "+str(i+1))

                # Perform the optimization using Nelder-Mead
                result = minimize(cost_func, 
                                initial_guess, 
                                args = (gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, verbose, plot), 
                                method='Nelder-Mead',
                                options={'disp': True,
                                        # 'fatol': 1e-6, # TODO: Doesn't work
                                        'xatol': 0.1,
                                        'initial_simplex': initial_simplex},
                                bounds = ((-180,180),(-90,90)))
                        
                print("GS FOUND, Location: "+str(result.x))
                print(result)
                gs_list.append(return_bdm_gs(result.x[0], result.x[1]))
                gs_list_plot.append([result.x[0], result.x[1]])
                
        return gs_list, gs_list_plot #, agg_list_of_simplexes


