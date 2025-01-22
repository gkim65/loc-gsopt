import numpy as np
from scipy.optimize import minimize
from common.objective_functions import cost_func_gap

# TODO: add the cyclic coordinate descent part
def nelder_mead_scipy(cfg,land_data,global_list_of_simplexes,epc_start,epc_end,satellites):

    # Initial guess for the coordinates (x, y) 
    # Overridden currently by initial simplex
    initial_guess = np.array([-30, -60])

    # TODO: Find best initial simplex to start, may need to change based on continents
    initial_simplex =  np.array([[-30, -60],[-100, -60],[-30, -90]])
    
    # Additional args for function
    gs_list = []
    sat_list = satellites[0:cfg.problem.sat_num]
    land_geometries = land_data['geometry']
    plot = False

    # Perform the optimization using Nelder-Mead
    result = minimize(cost_func_gap, 
                    initial_guess, 
                    args = (gs_list, global_list_of_simplexes, sat_list, epc_start, epc_end, land_geometries, plot), 
                    method='Nelder-Mead',
                    options={'disp': True,
                            'maxiter': 5,
                            'initial_simplex': initial_simplex},
                    bounds = ((-180,180),(-90,90)))
    