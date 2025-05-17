import brahe as bh
import brahe.data_models as bdm
from common.utils import load_earth_data,mp_compute_contact_times,contactExclusion # compute_earth_interior_angle

from common.sat_gen import satellites_from_constellation
from common.plotting import plot_gif,plot_img
# from methods.free_select.nelder_mead_scipy import nelder_mead_scipyfrom methods.free_select.nelder_mead_scipy import nelder_mead_scipy
from methods.free_select.scipy_methods import nelder_mead_scipy, powell_scipy
from methods.free_select.scipy_ccgs import nelder_mead_scipy_ccgs
from methods.free_select.genetic_algorithms import diffEvolution

from methods.teleport.ILP import ILP_Model
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
from common.sat_gen import make_tle
from common.station_gen import teleport_json
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

################################### Global Variables ###################################
# Usually global variables is bad practice, but just using it for land boundary dataset

# Load land boundary dataset
# using: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
land_data = gpd.read_file("data/ne_10m_admin_0_countries.shp")  

################################### Main Script ###################################

# Bringing in Hydra configuration parameters
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    
    ########## Configuration files: ##########
    #Ensure unique project names every time
    proj_name = cfg.problem.type+"_"+cfg.problem.method+"_"+cfg.problem.objective+"_"+str(cfg.problem.gs_num)+"_"+str(cfg.problem.sat_num)
    scenario_name = cfg.scenario.constellations
    constraints_name = str(cfg.constraints.dist_other_gs)

    if cfg.debug.wandb:
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.test_name+proj_name+"="+scenario_name+"="+constraints_name)

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
    
    # Make sure to load in earth inertial data every start time!
    load_earth_data('data/iau2000A_finals_ab.txt',cfg.debug.txtUpdate)

    satellites = satellites_from_constellation(cfg.scenario.constellations, cfg.debug.txtUpdate)[0:cfg.problem.sat_num]

    ########## Solvers: ##########

    # TODO: add other methods
    if cfg.problem.type == "free":
        if cfg.problem.method == "nelder":
            
            gs_list,  gs_list_plot = nelder_mead_scipy(cfg,land_data,epc_start,epc_end,satellites) # agg_list_of_simplexes
            
            if cfg.debug.wandb:
                contacts, _ = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, False)
                _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                run.summary["gs_list"] = gs_list_plot 
                run.summary["contact_num"] = len(contacts_exclusion_secs) 
                run.summary["seconds"] = np.sum(contacts_exclusion_secs)
                run.summary["data_downlink"] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate

                wandb.save("data/*")

                if cfg.debug.plot:
                    print("##############################")
                    print(gs_list_plot)
                    plot_img(gs_list_plot,"gs_all.png")
                    print(wandb.Image(plot_img(gs_list_plot,"gs_all.png")))
                    figure = wandb.Image(plot_img(gs_list_plot,"gs_all.png"))
                    run.log({"gs_all": figure})
        
        if cfg.problem.method == "nelder_ccgs":
            
            gs_list,  gs_list_plot = nelder_mead_scipy_ccgs(cfg,land_data,epc_start,epc_end,satellites) # agg_list_of_simplexes
            
            if cfg.debug.wandb:
                contacts, _ = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, False)
                _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                run.summary["gs_list"] = gs_list_plot 
                run.summary["contact_num"] = len(contacts_exclusion_secs) 
                run.summary["seconds"] = np.sum(contacts_exclusion_secs)
                run.summary["data_downlink"] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate

                wandb.save("data/*")

                if cfg.debug.plot:
                    print("##############################")
                    print(gs_list_plot)
                    plot_img(gs_list_plot,"gs_all.png")
                    print(wandb.Image(plot_img(gs_list_plot,"gs_all.png")))
                    figure = wandb.Image(plot_img(gs_list_plot,"gs_all.png"))
                    run.log({"gs_all": figure})


        if cfg.problem.method == "powell":
            
            gs_list,  gs_list_plot = powell_scipy(cfg,land_data,epc_start,epc_end,satellites) # agg_list_of_simplexes
            
            if cfg.debug.wandb:
                contacts, _ = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, False)
                _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                run.summary["gs_list"] = gs_list_plot 
                run.summary["contact_num"] = len(contacts_exclusion_secs) 
                run.summary["seconds"] = np.sum(contacts_exclusion_secs)
                run.summary["data_downlink"] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate

                wandb.save("data/*")

                if cfg.debug.plot:
                    print("##############################")
                    print(gs_list_plot)
                    plot_img(gs_list_plot,"gs_all.png")
                    print(wandb.Image(plot_img(gs_list_plot,"gs_all.png")))
                    figure = wandb.Image(plot_img(gs_list_plot,"gs_all.png"))
                    run.log({"gs_all": figure})

        if cfg.problem.method == "diffEvolution":

            gs_list,  gs_list_plot = diffEvolution(cfg,land_data,epc_start,epc_end,satellites) # agg_list_of_simplexes
            
            if cfg.debug.wandb:
                contacts, _ = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, False)
                _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                run.summary["gs_list"] = gs_list_plot 
                run.summary["contact_num"] = len(contacts_exclusion_secs) 
                run.summary["seconds"] = np.sum(contacts_exclusion_secs)
                run.summary["data_downlink"] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate

                wandb.save("data/*")

                if cfg.debug.plot:
                    print("##############################")
                    print(gs_list_plot)
                    plot_img(gs_list_plot,"gs_all.png")
                    print(wandb.Image(plot_img(gs_list_plot,"gs_all.png")))
                    figure = wandb.Image(plot_img(gs_list_plot,"gs_all.png"))
                    run.log({"gs_all": figure})

    elif cfg.problem.type == "teleport":
        #Load ground stations from JSON file
        ground_stations = teleport_json('data/test_teleports.json')[0:cfg.problem.teleport_num]
        print(f"Loaded {len(ground_stations)} ground stations")

        ilp_model = ILP_Model(ground_stations, satellites, epc_start, epc_end, cfg.problem.gs_num, data_rate=cfg.problem.data_rate)
        
        if cfg.problem.objective == "data_downlink":
            selected_stations, station_contacts, output_data = ilp_model.data_downlink_ilp()
            #output data is total data downlinked
        
        if cfg.problem.objective == "gap_optimization":
            selected_stations, station_contacts, output_data = ilp_model.gap_time_ilp()
            #output data is gap times
        
        if cfg.problem.objective == "max_contacts":
            selected_stations, station_contacts, output_data = ilp_model.max_contact_ilp()
            #output data is contact count
        
        if cfg.debug.wandb:
            run.summary["selected_stations"] = selected_stations
            run.summary["output_data"] = output_data
            run.summary["lat_long"] = [[gs.geometry.coordinates[0], gs.geometry.coordinates[1]] for gs in [ground_stations[i] for i in selected_stations]]
            run.summary["gs_list"] = [station_contacts[i]['name'] for i in selected_stations]
            artifact = wandb.Artifact("station_contacts", type="json")     
            run.log_artifact(artifact)  
        
        if cfg.debug.plot == True:
            # Get the actual ground station objects for the selected stations
            selected_gs = [ground_stations[i] for i in selected_stations]
            
            figure = wandb.Image(plot_gap_times(satellites, selected_gs, epc_start, epc_end, cfg.debug.plot))
            run.log({"gap_times": figure})

    if cfg.debug.wandb:
        run.finish()
        wandb.finish()



if __name__ == "__main__":
    main()