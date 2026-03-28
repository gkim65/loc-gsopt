import brahe as bh
# import brahe.data_models as bdm
from common.utils import load_earth_data,compute_contact_times,contactExclusion

from common.sat_gen import satellites_from_constellation
from common.plotting import plot_gif,plot_img
# from methods.free_select.nelder_mead_scipy import nelder_mead_scipyfrom methods.free_select.nelder_mead_scipy import nelder_mead_scipy
from methods.free_select.scipy_methods import nelder_mead_scipy, powell_scipy
from methods.free_select.scipy_ccgs import nelder_mead_scipy_ccgs
from methods.free_select.genetic_algorithms import diffEvolution
# from methods.teleport.ILP import ILP_Model
from common.plotting import plot_contact_windows, plot_gap_times
###Vedant's imports
# Standard imports
import sys
import os
from itertools import groupby
import pyomo.environ as pyo
import pyomo.kernel as pk

# Add the path to the folder containing the module
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Required imports
from common.sat_gen import make_walker_constellation
from common.station_gen import gs_json, gs_json_list
from common.utils import load_earth_data, gap_times_condense
#####

import numpy as np

# Hydra
import hydra
from omegaconf import DictConfig
import omegaconf

# WandB
import wandb

# Shape files
import geopandas as gpd

from datetime import timedelta
import datetime

################################### Global Variables ###################################
# Usually global variables is bad practice, but just using it for land boundary dataset

# Load land boundary dataset
# using: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
land_data = gpd.read_file("data/ne_10m_admin_0_countries.shp")  

################################### Main Script ###################################

# Bringing in Hydra configuration parameters
@hydra.main(version_base=None, config_path="../config", config_name="config_surrogate")
def main(cfg: DictConfig):
    
    ########## Configuration files: ##########
    #Ensure unique project names every time
    proj_name = cfg.problem.type+"_"+cfg.problem.method+"_"+cfg.problem.objective+"_"+str(cfg.problem.gs_num)+"_"+str(cfg.problem.sat_num)
    scenario_name = cfg.scenario.constellations
    constraints_name = str(cfg.constraints.dist_other_gs)

    if cfg.debug.wandb:
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.test_name, reinit=True)

    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    if cfg.debug.verbose:  
        print(type(config_dict), config_dict)

    if cfg.debug.wandb:
        wandb.config.update(config_dict, allow_val_change=True)

    ########## Initial Scenario Setup: ##########

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
    
    # Set random seed
    np.random.seed(cfg.debug.randseed)
    
    eop_file_custom = bh.FileEOPProvider.from_standard_file(
        "data/iau2000A_finals_ab.txt",  # Replace with actual file path
        True,  # Interpolation
        "Hold",  # Extrapolation
    )
    bh.set_global_eop_provider(eop_file_custom)
    # Make sure to load in earth inertial data every start time!
    # load_earth_data('data/iau2000A_finals_ab.txt',cfg.debug.txtUpdate)

    if cfg.walker.custom:
        satellites = make_walker_constellation(
            epoch=epc_start,
            altitude_km=cfg.walker.altitude,
            eccentricity=cfg.walker.eccentricity,
            inclination=cfg.walker.inclination,
            num_planes=4,#cfg.walker.num_planes,
            sats_per_plane= cfg.walker.sats_perplane,
            phase=cfg.walker.phase,
            argp=cfg.walker.argp,
            norad_start=cfg.walker.norad_start,
            star =cfg.walker.star,
        )
    else:
        satellites = satellites_from_constellation(cfg.scenario.constellations, cfg.debug.txtUpdate)[0:cfg.problem.sat_num]
    print(satellites)


    # Step 1: Propagate all satellites in parallel
    bh.par_propagate_to(satellites, epc_end)
    ########## Solvers: ##########


    # -12.261594786655923,-89.99634831100451
    #-22.64991657456346,82.83197826524209
    runs = [
        # Set 1
        [[15.65, 78.23],      # Svalbard, Norway (78°N)
                [2.53, -72.01],      # Troll Station, Antarctica (72°S)
                [-133.72, 68.36],    # Inuvik, Canada (68°N)
                [-57.85, -51.68]],    # Stanley, Falkland Islands (52°S)

        # Set 2
        [[-26.51, 64.14],     # Reykjavik, Iceland (64°N)
                [2.53, -72.01],      # Troll Station, Antarctica (72°S)
                [-148.49, 70.26],    # Prudhoe Bay, Alaska (70°N)
                [168.38, -46.53]],    # Awarua, New Zealand (47°S)

        # Set 3
        [[25.75, 71.17],      # Vardo, Norway (71°N)
                [2.53, -72.01],      # Troll Station, Antarctica (72°S)
                [-51.72, 64.18],     # Nuuk, Greenland (64°N)
                [-70.87, -52.94]],
        [[31.11, 70.37],      # Vardo, Norway (70°N)
           [2.53, -72.01],      # Troll Station, Antarctica (72°S)
           [-68.70, 76.53],     # Baffin Island, Canada (77°N)
           [51.73, -46.28]],
        [[-68.70, 76.53],     # Baffin Island, Canada (77°N)
        [2.53, -72.01],      # Troll Station, Antarctica (72°S)
        [25.75, 71.17],      # Vardo, Norway (71°N)
        [-57.85, -51.68]],   # Stanley, Falkland Islands (52°S)
]     # Alfred Faure, Crozet (46°S)    # Punta Arenas, Chile (53°S)


    # Set 5: Maximum longitudinal diversity
    list_gs = runs[4]      # Troll Station, Antarctica
    gs_list = []
    for i, new_gs in enumerate(list_gs):
        point_loc = bh.PointLocation(new_gs[0], new_gs[1])
        point_loc.set_id(i)
        gs_list.append(point_loc)
    gs_list_plot = [-12.261594786655923,-89.99634831100451]
    
    contacts, contact_secs = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
    _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
    print(np.sum(contact_secs)*cfg.scenario.datarate*52/1000/1000/1000/1000/1000/8)
    print(np.sum(contacts_exclusion_secs)*cfg.scenario.datarate*52/1000/1000/1000/1000/1000/8)
    if cfg.debug.wandb:
        run.log({"gs_list_lat": gs_list_plot[0],
            "gs_list_long": gs_list_plot[1],
            "total_contact_num": len(contacts_exclusion_secs) ,
            "total_seconds": np.sum(contacts_exclusion_secs),
            "altitude": cfg.walker.altitude,
            "inclination":cfg.walker.inclination,
            "year": cfg.end_epoch.year, 
            "month" : cfg.end_epoch.month, 
            "day": cfg.end_epoch.day})
        run.summary["gs_list"] = gs_list_plot 
        run.summary["contact_num"] = len(contacts_exclusion_secs) 
        run.summary["seconds"] = np.sum(contacts_exclusion_secs)
        run.summary["data_downlink"] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate

        wandb.save("data/*")

    if cfg.debug.wandb:
        run.finish()
        wandb.finish()



if __name__ == "__main__":
    main()