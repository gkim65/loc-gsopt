import numpy as np
from scipy.optimize import minimize
from common.objective_functions import cost_func
from common.station_gen import return_bdm_gs
from common.utils import mp_compute_contact_times

#  WandB
import wandb
import copy


# TODO: add the cyclic coordinate descent part (May not include in this paper)
def nelder_mead_scipy(cfg,land_data,epc_start,epc_end,satellites):
        

        # Setup args for minimize function
        gs_list = []
        gs_list_plot = []
        gs_contacts_og = []
        
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
                
                # initial_simplex =
                initial_simplex_list = [ np.array([
                        [-60, -85],  # Point 1 (near the Caribbean)
                        [151.2093, -33.8688],  # Point 2 (Sydney, Australia)
                        [-153.67891140271288, 55.17076207063536]  # Point 3 (near Alaska)
                        ]),
                np.array([
                        [37.7749, -122.4194],  # Point 1 (San Francisco, USA)
                        [-33.8688, 151.2093],  # Point 2 (Sydney, Australia)
                        [55.7558, 37.6176]     # Point 3 (Moscow, Russia)
                ]),
                
                np.array([
                        [40.7128, -74.0060],   # Point 1 (New York City, USA)
                        [-22.9068, -43.1729],  # Point 2 (Rio de Janeiro, Brazil)
                        [34.0522, -118.2437]   # Point 3 (Los Angeles, USA)
                ]),

                np.array([
                        [51.5074, -0.1278],    # Point 1 (London, UK)
                        [-34.6037, -58.3816],  # Point 2 (Buenos Aires, Argentina)
                        [35.6895, 139.6917]    # Point 3 (Tokyo, Japan)
                ]),

                np.array([
                        [45.4215, -75.6972],   # Point 1 (Ottawa, Canada)
                        [39.9042, 116.4074],   # Point 2 (Beijing, China)
                        [-37.8136, 144.9631]   # Point 3 (Melbourne, Australia)
                ]),

                np.array([
                        [-1.286389, 36.817223],  # Point 1 (Nairobi, Kenya)
                        [40.730610, -73.935242], # Point 2 (New York City, USA)
                        [39.0742, 21.8243]       # Point 3 (Athens, Greece)
                ])
                ]

                print(initial_simplex_list[cfg.scenario.start_simplex])

                
                print("STARTING TO PERFORM MINIMIZATION ON GS: "+str(i+1))

                # Perform the optimization using Nelder-Mead
                result = minimize(cost_func, 
                                initial_guess, 
                                args = (gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og, verbose, plot), 
                                method='Nelder-Mead',
                                options={'disp': True,
                                        # 'fatol': 1e-6, # TODO: Doesn't work
                                        # 'xatol': 3, # this might be too much! # TODO: Doesn't work
                                        'maxiter': 25, # this might be too little!
                                        'initial_simplex': initial_simplex_list[cfg.scenario.start_simplex]},
                                bounds = ((-180,180),(-90,90)))
                        
                print("GS FOUND, Location: "+str(result.x))
                print(result)
                gs_list.append(return_bdm_gs(result.x[0], result.x[1]))
                gs_list_plot.append([result.x[0], result.x[1]])

                # try to minimize number of contacts to compute:
                contacts_og, _ = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, plot)
                gs_contacts_og = contacts_og
                
        return gs_list, gs_list_plot 


