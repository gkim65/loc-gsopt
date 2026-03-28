import wandb
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib as mpl

from matplotlib.patches import Rectangle

# Hydra
import hydra
from omegaconf import DictConfig
import omegaconf

# WandB
import wandb

import brahe as bh
from common.sat_gen import satellites_from_constellation
from common.utils import compute_contact_times,contactExclusion
import os

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size" : 24, 
    "font.serif": ["Computer Modern Roman"],  # optional: you can specify others like Times
    "axes.unicode_minus": False  # optional: fix minus signs in LaTeX
})

# Bringing in Hydra configuration parameters
@hydra.main(version_base=None, config_path="../config", config_name="config_linePlot")
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
    
    eop_file_custom = bh.FileEOPProvider.from_standard_file(
        "data/iau2000A_finals_ab.txt",  # Replace with actual file path
        True,  # Interpolation
        "Hold",  # Extrapolation
    )
    bh.set_global_eop_provider(eop_file_custom)

    satellites = satellites_from_constellation(cfg.scenario.constellations, False)
    
    if cfg.scenario.constellations == "CAPELLA":
        constellation = "CAPELLA Space"
    if cfg.scenario.constellations == "ICEYE":
        constellation = "ICEYE"
    

    
    # Step 1: Propagate all satellites in parallel
    bh.par_propagate_to(satellites, epc_end)
    
    ############################# Prepping Data ###############################

    try:
        e_data_downlink_df = pd.read_csv(f'nelder_{cfg.scenario.constellations}.csv')
    except FileNotFoundError:
        e_data_downlink_df = df_generatorNelder("loc_gsopt/ccgs_betterfree_nelder_ccgs_data_downlink",constellation,satellites, epc_end, epc_start, cfg)
        e_data_downlink_df.to_csv(f'nelder_{cfg.scenario.constellations}.csv', index=False)  
    idx = e_data_downlink_df.groupby("gs_number")["data_downlinked"].idxmax()
    max_data_downlink = e_data_downlink_df.groupby("gs_number")["data_downlinked"].max().reset_index()

    try:
        e_lat_data_downlink_df = pd.read_csv(f'nelder_lat_{cfg.scenario.constellations}.csv')
    except FileNotFoundError:
        e_lat_data_downlink_df = df_generatorNelder("loc_gsopt/SCORE_LATfree_nelder_ccgs_data_downlink",constellation,satellites, epc_end, epc_start, cfg)
        e_lat_data_downlink_df.to_csv(f'nelder_lat_{cfg.scenario.constellations}.csv', index=False)  
    idx = e_lat_data_downlink_df.groupby("gs_number")["data_downlinked"].idxmax()
    max_lat_data_downlink = e_lat_data_downlink_df.groupby("gs_number")["data_downlinked"].max().reset_index()


    teleport_data_downlink_df = df_generatorTeleport(constellation)
    tele_max_data_downlink = teleport_data_downlink_df.groupby("gs_number")["data_downlinked"].max().reset_index()

    ksat_data_downlink_df = df_generatorKSAT(constellation)


    ############################# Plotting Figures ###############################

    plt.figure(figsize=(10, 8))


    # Plot the line graph
    plt.plot(
        max_data_downlink["gs_number"]+1, 
        max_data_downlink["data_downlinked"]/1000/1000*52, 
        color='blue', 
        marker='o', 
        markersize=4,  # Smaller marker size
        linestyle=':', 
        label='Max Data Downlinked'
    )
    plt.plot(
        max_lat_data_downlink["gs_number"]+1, 
        max_lat_data_downlink["data_downlinked"]/1000/1000*52, 
        color='cornflowerblue', 
        marker='D', 
        markersize=4,  # Smaller marker size
        linestyle='-.', 
        label='Max Data Downlinked'
    )

    e_data_downlink_df
    plt.plot(
        tele_max_data_downlink["gs_number"], 
        tele_max_data_downlink["data_downlinked"]/1000/1000*52, 
        color='purple', 
        marker='^', 
        markersize=4,  # Smaller marker size
        linestyle='--', 
        label='Teleports'
    )

    # Create a line plot for ksat_data_downlink_df
    plt.plot(
        ksat_data_downlink_df["gs_number"], 
        ksat_data_downlink_df["data_downlinked"]/1000/1000*52, 
        color='red', 
        marker='s', 
        markersize=4,  # Smaller marker size
        linestyle='-', 
        label='KSAT'
    )

    # plt.title("Ground Station Selections of Free Select V.S. Fixed Location Methods ")
    plt.xlabel(f"Ground Stations Selected for {constellation} Constellation")
    plt.ylabel(r"Total Data Downlinked (PB) over $T_{opt}$")
    plt.legend(
        handles=[
            # plt.Line2D([0], [0], marker='o', color='black', markersize=8, linestyle='None', label='Nelder Mead'), 
            plt.Line2D([0], [0], marker='o', color='blue', markersize=8, linestyle=':', label='SCORE'), 
            plt.Line2D([0], [0], marker='2', color='blue', markersize=8, linestyle=':', label='SCORE-Constraint'), 
            plt.Line2D([0], [0], marker='^', color='purple', markersize=8, linestyle='--', label='Teleports'),
            plt.Line2D([0], [0], marker='s', color='red', markersize=8, linestyle='-', label='KSAT')
        ],
        loc='lower right'
    )

    x_ticks = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    plt.xticks(ticks=x_ticks, labels=x_ticks)


    # Add shaded rectangle for zoomed region around gs_number = 2
    if constellation == "CAPELLA Space":
        plt.gca().add_patch(
            Rectangle(
                (1.5, 160/1000*52),        # (x, y) bottom-left corner
                1,                  # width
                40/1000*52,                # height
                linewidth=1,
                edgecolor='gray',
                facecolor='lightgray',
                alpha=0.5,
                zorder=0
            )
        )
    else:
        plt.gca().add_patch(
                Rectangle(
                    (1.5, 750/1000*52),        # (x, y) bottom-left corner
                    1,                  # width
                    1100/1000*52-750/1000*52,                 # height

                    linewidth=1,
                    edgecolor='gray',
                    facecolor='lightgray',
                    alpha=0.5,
                    zorder=0
                )
            )
        
    plt.tight_layout()
    # plt.ylim(80000,100000)
    # plt.yscale("log")
    # plt.xlim(0,5)
    os.makedirs("figures_final", exist_ok=True)
    if constellation == "CAPELLA Space":
        plt.savefig("figures_final/capellaDataDownlink.pdf")
    if constellation == "ICEYE":
        plt.savefig("figures_final/iceyeDataDownlink.pdf")

    plt.show()


############################# Functions for Prepping Data ###############################

def df_generatorNelder(name1,constellation,satellites, epc_end, epc_start,cfg):

    api = wandb.Api()
    
    e_data_downlink_list = []
    
    # Iterate through the ground station numbers and runs
    if constellation == "ICEYE":
        text = "34=ICEYE" 
    if constellation == "CAPELLA Space":
        text = "5=CAPELLA"
    
    for i in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        project_name = f"{name1}_{i}_{text}=3000000"
        runs = api.runs(project_name)
        for run in runs:
            print(run)
            if run.state == "finished":
                # Append each run's data as a dictionary to the list

                # if constellation != "CAPELLA Space":
                gs_list = []
                for i, new_gs in enumerate(run.summary['gs_list']):
                    point_loc = bh.PointLocation(new_gs[0], new_gs[1])
                    point_loc.set_id(i)
                    gs_list.append(point_loc)

                contacts, contact_secs = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                print(sum(contact_secs))
                print(sum(contacts_exclusion_secs))
                e_data_downlink_list.append({
                    "gs_number": i*1.0, #+0.25
                    "data_downlinked": np.sum(contacts_exclusion_secs)*cfg.scenario.datarate/1000000000,
                    "gs_list": run.summary.gs_list
                    })
                # else:
                #     e_data_downlink_list.append({
                #         "gs_number": i*1.0, #+0.25
                #         "data_downlinked": run.summary.data_downlink/1000000000,
                #         "gs_list": run.summary.gs_list
                #     })

    # Convert the list of dictionaries into a pandas DataFrame
    e_data_downlink_df = pd.DataFrame(e_data_downlink_list)
    return e_data_downlink_df

def df_generatorTeleport(constellation):
    api = wandb.Api()

    teleport_data_downlink_list = []

    # Iterate through the ground station numbers and runs
    if constellation == "ICEYE":
        text = "34=ICEYE" 
    if constellation == "CAPELLA Space":
        text = "5=CAPELLA"

    # Iterate through the ground station numbers and runs
    if constellation == "CAPELLA Space":
        gs_iterate = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    else:
        gs_iterate = [1, 2, 3, 5, 7, 10, 15, 20] # take out 4 if needed

    for i in gs_iterate:
        # project_name = f"loc_gsopt/teleport_select_100_data_downlink_{i}_5=CAPELLA=3000000"
        project_name = f"loc_gsopt/teleport_select_100_data_downlink_{i}_{text}=3000000"
        runs = api.runs(project_name)
        try:
            for run in runs:
                if run.state == "finished":
                    # Append each run's data as a dictionary to the list
                    teleport_data_downlink_list.append({
                        "gs_number": i*1.0,
                        "data_downlinked": run.summary.output_data/1000000000
                    })
        except:
            print("not found")

    # Convert the list of dictionaries into a pandas DataFrame
    teleport_data_downlink_df = pd.DataFrame(teleport_data_downlink_list)
    return teleport_data_downlink_df

def df_generatorKSAT(constellation):
    api = wandb.Api()

    ksat_data_downlink_list = []

    # Iterate through the ground station numbers and runs
    if constellation == "ICEYE":
        text = "34=ICEYE" 
    if constellation == "CAPELLA Space":
        text = "5=CAPELLA"

    # Iterate through the ground station numbers and runs
    for i in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        project_name = f"loc_gsopt/KSAT_MILP_data_downlink_{i}_{text}=3000000"
        try:
            runs = api.runs(project_name)
            for run in runs:
                if run.state == "finished":
                    print(run)
                    # Append each run's data as a dictionary to the list
                    ksat_data_downlink_list.append({
                        "gs_number": i*1.0,
                        "data_downlinked": run.summary["data_downlinked.total"]/1000000000
                    })
        except:
            print("notfound")

    # Convert the list of dictionaries into a pandas DataFrame
    ksat_data_downlink_df = pd.DataFrame(ksat_data_downlink_list)
    return ksat_data_downlink_df


if __name__ == "__main__":
    main()