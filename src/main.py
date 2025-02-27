import brahe as bh
from common.utils import load_earth_data #,compute_all_gaps_contacts, compute_earth_interior_angle
from common.sat_gen import satellites_from_constellation
from common.plotting import plot_gif,plot_img
from methods.free_select.nelder_mead_scipy import nelder_mead_scipy

# TODO: REmove DEBUGGING:
from common.utils import compute_gaps_per_sat

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

    # print(cfg.scenario.constellations)
    # print(len(satellites))
    # for i in satellites:
    #     print(i)

    # TODO: add other methods
    if cfg.problem.type == "free":
        if cfg.problem.method == "nelder":
            
            gs_list,  gs_list_plot, agg_list_of_simplexes = nelder_mead_scipy(cfg,land_data,global_list_of_simplexes,epc_start,epc_end,satellites)

            for ind, simplexes in enumerate(agg_list_of_simplexes):
                plot_gif(simplexes,"gs_"+str(ind+1)+".gif",gs_list=gs_list_plot[0:ind],ind=ind)

            print("##############################")
            print(gs_list_plot)
            # print(agg_list_of_simplexes)
            plot_img(gs_list_plot,"gs_all.png")

            # DEBUGG:
            # compute_gaps_per_sat(satellites,gs_list_plot,epc_start,epc_end, True,"gap_times_chart.png")


if __name__ == "__main__":
    main()