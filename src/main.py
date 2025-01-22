import brahe as bh
from common.utils import load_earth_data #,compute_all_gaps_contacts, compute_earth_interior_angle
from common.sat_gen import satellites_from_constellation
from common.objective_functions import cost_func_gap
from common.plotting import plot_gif

from scipy.optimize import minimize
import numpy as np

# Hydra
import hydra
from omegaconf import DictConfig

# Shape files
import geopandas as gpd

################################### Global Variables ###################################
# Usually global variables is bad practice, but more for plotting purposes :D

# Load land boundary dataset
# using: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
land_data = gpd.read_file("data/ne_10m_admin_0_countries.shp")  

global_list_of_simplexes = [] # For plotting purposes

################################### Main Script ###################################

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    # Setting up start and end epochs
    epc_start = bh.Epoch(cfg.start_epoch.year, 
                         cfg.start_epoch.month, 
                         cfg.start_epoch.day, 
                         cfg.start_epoch.hour, 
                         cfg.start_epoch.minute, 
                         cfg.start_epoch.second) 
    
    epc_end = bh.Epoch(cfg.end_epoch.year, 
                         cfg.end_epoch.month, 
                         cfg.end_epoch.day, 
                         cfg.end_epoch.hour, 
                         cfg.end_epoch.minute, 
                         cfg.end_epoch.second) 
    
    # Make sure to load in earth inertial data every start time!
    load_earth_data('data/iau2000A_finals_ab.txt')

    satellites = satellites_from_constellation(cfg.scenario.constellations)

    # TODO: add other methods
    if cfg.problem.type == "free":
        if cfg.problem.method == "nelder":
    
            # Initial guess for the coordinates (x, y) 
            # Overridden currently by initial simplex
            initial_guess = np.array([-30, -60])

            # TODO: Find best initial simplex to start, may need to change based on continents
            initial_simplex =  np.array([[-30, -60],[-100, -60],[-30, -90]])
            
            # Additional args for function
            gs_list = []
            sat_list = satellites=satellites[0:cfg.problem.sat_num]
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
            
            print(result)
            plot_gif(global_list_of_simplexes)
            print(global_list_of_simplexes)

if __name__ == "__main__":
    main()