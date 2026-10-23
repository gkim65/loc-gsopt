import brahe as bh
from common.objective_functions import EvalCounter
from common.utils import compute_contact_times,contactExclusion
from common.sat_gen import satellites_from_constellation, make_walker_constellation
from common.station_gen import teleport_json
from common.plotting import plot_img, plot_gap_times
from methods.free_select.scipy_ccgs import nelder_mead_scipy_ccgs, powell_scipy
from methods.free_select.genetic_algorithms import diffEvolution
# from methods.teleport.ILP import ILP_Model

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
city_data = gpd.read_file("data/city_buffers.shp")

eval_counter = EvalCounter()      # single global instance

# Options whose code is kept but cannot currently run, with the reason shown to the user
UNSUPPORTED = {
    ("objective", "gap_optimization"): "gap-time computation uses the pre-1.0 brahe access API",
    ("objective", "max_contacts"): "contact exclusion has no contact-count objective",
    ("type", "teleport"): "the MILP teleport-selection model uses the pre-1.0 brahe access API; "
                          "its paper results are in data/csv_Files/milp_*.csv",
}


def check_supported(cfg):
    """
        check_supported(cfg) -> None

    Raise before any computation if the config selects an option that cannot currently run.

      - `cfg` — the Hydra config for this run

    Raises `NotImplementedError` naming the option and the reason it is unavailable.
    """
    for (key, value), reason in UNSUPPORTED.items():
        if cfg.problem[key] == value:
            raise NotImplementedError(f'problem.{key}="{value}" is not currently supported: {reason}.')

################################### Main Script ###################################

# Bringing in Hydra configuration parameters
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    check_supported(cfg)

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
    np.random.seed(int(cfg.debug.randseed))
    
    # Make sure to load in earth inertial data every start time!
    # load_earth_data('data/iau2000A_finals_ab.txt',cfg.debug.txtUpdate)
    eop_file_custom = bh.FileEOPProvider.from_standard_file(
        "data/iau2000A_finals_ab.txt",  # Replace with actual file path
        True,  # Interpolation
        "Hold",  # Extrapolation
    )
    bh.set_global_eop_provider(eop_file_custom)

    if cfg.walker.custom:
        satellites = make_walker_constellation(
            epoch=epc_start,
            altitude_km=cfg.walker.altitude,
            eccentricity=cfg.walker.eccentricity,
            inclination=cfg.walker.inclination,
            num_planes=cfg.walker.num_planes,
            sats_per_plane= cfg.walker.sats_perplane,
            phase=cfg.walker.phase,
            argp=cfg.walker.argp,
            norad_start=cfg.walker.norad_start,
            star =cfg.walker.star,
        )
    else:
        satellites = satellites_from_constellation(cfg.scenario.constellations, cfg.debug.txtUpdate)[0:cfg.problem.sat_num]

    # Step 1: Propagate all satellites in parallel
    bh.par_propagate_to(satellites, epc_end)
    ########## Solvers: ##########

    # TODO: add other methods
    if cfg.problem.type == "free":
        if cfg.problem.method == "nelder_ccgs":
            

            seed = cfg.debug.randseed  # e.g., 42, 123, 7, ...
            rng = np.random.default_rng(seed) if cfg.experiments.ccgs_random else None

            gs_list,  gs_list_plot = nelder_mead_scipy_ccgs(cfg,land_data,epc_start,epc_end,satellites,eval_counter,city_data, rng=rng) # agg_list_of_simplexes
            
            if cfg.debug.wandb:
                contacts, _ = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
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
            
            gs_list,  gs_list_plot = powell_scipy(cfg,land_data,epc_start,epc_end,satellites,eval_counter,city_data) # agg_list_of_simplexes
            
            if cfg.debug.wandb:
                contacts, _ = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
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

            gs_list,  gs_list_plot = diffEvolution(cfg,land_data,epc_start,epc_end,satellites,eval_counter,city_data) # agg_list_of_simplexes
            
            if cfg.debug.wandb:
                contacts, _ = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
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
        ground_stations = teleport_json('data/most_diverse_100_ground_stations.json')[0:cfg.problem.teleport_num]
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