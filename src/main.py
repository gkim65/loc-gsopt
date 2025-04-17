import brahe as bh
from common.utils import load_earth_data,mp_compute_contact_times,contactExclusion # compute_earth_interior_angle
from common.sat_gen import satellites_from_constellation
from common.plotting import plot_gif,plot_img
from methods.free_select.nelder_mead_scipy import nelder_mead_scipy


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
        run = wandb.init(entity=cfg.wandb.entity, project=proj_name+"="+scenario_name+"="+constraints_name)

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

    satellites = satellites_from_constellation(cfg.scenario.constellations, cfg.debug.txtUpdate)


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
                    figure = wandb.Image(plot_img(gs_list_plot,f"gs_all.png"))
                    run.log({"gs_all": figure})

    if cfg.debug.wandb:
        run.finish()



if __name__ == "__main__":
    main()