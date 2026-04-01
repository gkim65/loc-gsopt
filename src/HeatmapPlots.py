import wandb
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib as mpl
from matplotlib.colors import Normalize

from matplotlib.patches import Rectangle

# Hydra
import hydra
from omegaconf import DictConfig
import omegaconf

# WandB
import wandb

import brahe as bh
from common.sat_gen import make_walker_constellation
from common.utils import compute_contact_times,contactExclusion
import os




mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size" : 24, 
    "font.serif": ["Computer Modern Roman"],  # optional: you can specify others like Times
    "axes.unicode_minus": False  # optional: fix minus signs in LaTeX
})

from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import FuncFormatter

# Bringing in Hydra configuration parameters
@hydra.main(version_base=None, config_path="../config", config_name="config_surrogate")
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
    
    ############################# Prepping Data ###############################
    # data_downlink_df_powell = df_generatorPowell(epc_end, epc_start,cfg)
    # avg_data_downlink_df_score_all = df_generatorNelder(epc_end, epc_start,cfg)

    # heatmap_data_SCORE = avg_data_downlink_df_score_all.pivot(index='gs_number', columns='sats', values='WallClockTime')
    # # heatmap_data_DE = avg_data_downlink_df_all.pivot(index='gs_number', columns='sats', values='WallClockTime')# 'WallClockTime')
    # heatmap_data_DE = data_downlink_df_powell.pivot(index='gs_number', columns='sats', values='WallClockTime')

    data_downlink_df_powell = df_generator("heatmap1free_powell_data_downlink","data/csv_Files/powell_heatmap.csv",epc_end, epc_start,cfg)
    data_downlink_df_nelder = df_generator("heatmapfree_nelder_ccgs_data_downlink","data/csv_Files/nelder_heatmap.csv",epc_end, epc_start,cfg)
    data_downlink_df_DE = df_generator("heatmapfree_diffEvolution_data_downlink", "data/csv_Files/DE_heatmap.csv", epc_end, epc_start,cfg)
    heatmap_data_SCORE = data_downlink_df_nelder.pivot(index='gs_number', columns='sats', values='eval_counter4')/1000
    heatmap_data_powell = data_downlink_df_powell.pivot(index='gs_number', columns='sats', values='eval_counter4')/1000
    heatmap_data_DE = data_downlink_df_DE.pivot(index='gs_number', columns='sats', values='eval_counter')/1000


    # Compute global vmin and vmax for log scale
    vmin = min(heatmap_data_SCORE.values.min(), heatmap_data_powell.values.min(), heatmap_data_DE.values.min())
    vmax = max(heatmap_data_SCORE.values.max(), heatmap_data_powell.values.max(), heatmap_data_DE.values.max())

    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    sns.heatmap(heatmap_data_SCORE, ax=axes[0], annot=True, cmap='viridis', norm=norm, cbar=False)
    axes[0].set_title("SCORE (Nelder-Mead)")
    axes[0].set_xlabel("Number of Satellites")
    axes[0].set_ylabel("Number of Ground Stations")

    sns.heatmap(heatmap_data_powell, ax=axes[1], annot=True, cmap='viridis', norm=norm, cbar=False)
    axes[1].set_title("SCORE (Powell)")
    axes[1].set_xlabel("Number of Satellites")
    axes[1].set_ylabel("")

    sns.heatmap(heatmap_data_DE, ax=axes[2], annot=True, cmap="viridis", norm=norm, cbar=False)
    axes[2].set_title("Differential Evolution")
    axes[2].set_xlabel("Number of Satellites")
    axes[2].set_ylabel("")

    for ax in axes:
        ax.set_aspect('equal')  # forces square cells
    # Create a single colorbar for all heatmaps
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height] in figure coords
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # needed for ScalarMappable
    cbar = fig.colorbar(sm, cax=cbar_ax, format=LogFormatter())
    cbar.set_label(r"\# of Function Evaluations ($10^3$, log)")

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave room for colorbar
    plt.savefig("figures_final/heatmap_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    data_rate_bps = 1#1200000000.0  # bits per second

    # The Obj_func values are negative, so take absolute value for calculation
    # Obj_func is total seconds of contact, so total_data_bits = seconds * data_rate_bps
    # Convert bits to terabytes: 1 TB = 8e12 bits
    # Convert bits to megabytes: 1 MB = 8e6 bits
    # Since 'data_downlinked' is already in gigabytes (bits divided by 1e9), to convert to terabytes divide by 1000
    def gb_to_tb_opt(df):
        return df /8
    def obj_to_mb(df):
        return df.abs() * data_rate_bps / 8e6
    def obj_to_gb(df):
        return df.abs() * data_rate_bps / 8e9

    def obj_to_tb(df):
        return df.abs() * data_rate_bps / 8e12

    # Remove duplicate entries by taking the max Obj_func for each (gs_number, sats) pair
    # data_downlink_df_score_max = data_downlink_df_score.groupby(['gs_number', 'sats'], as_index=False).min(numeric_only=True)
    heatmap_data_obj_SCORE = data_downlink_df_nelder.pivot(index='gs_number', columns='sats', values='exclusion')#'Obj_func')
    heatmap_data_obj_powell = data_downlink_df_powell.pivot(index='gs_number', columns='sats', values= 'exclusion')#'FirstObj_func') #'Obj_func')#
    heatmap_data_obj_DE = data_downlink_df_DE.pivot(index='gs_number', columns='sats', values= 'exclusion')#'FirstObj_func') #'Obj_func')#

    heatmap_data_obj_SCORE = gb_to_tb_opt(heatmap_data_obj_SCORE)
    heatmap_data_obj_powell = gb_to_tb_opt(heatmap_data_obj_powell)
    heatmap_data_obj_DE = gb_to_tb_opt(heatmap_data_obj_DE)

    # Compute global vmin and vmax for log scale
    vmin = min(heatmap_data_obj_SCORE.values.min(), heatmap_data_obj_powell.values.min(), heatmap_data_obj_DE.values.min())
    vmax = max(heatmap_data_obj_SCORE.values.max(), heatmap_data_obj_powell.values.max(), heatmap_data_obj_DE.values.max())


    # Linear normalization
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    sns.heatmap(heatmap_data_obj_SCORE, ax=axes[0], annot=True, fmt='.1f', cmap='flare', norm=norm, cbar=False)
    axes[0].set_title("SCORE (Nelder-Mead)")
    axes[0].set_xlabel("Number of Satellites")
    axes[0].set_ylabel("Number of Ground Stations")

    sns.heatmap(heatmap_data_obj_powell, ax=axes[1], annot=True, fmt='.1f', cmap='flare', norm=norm, cbar=False)
    axes[1].set_title("SCORE (Powell)")
    axes[1].set_xlabel("Number of Satellites")
    axes[1].set_ylabel("")

    sns.heatmap(heatmap_data_obj_DE, ax=axes[2], annot=True, fmt='.1f', cmap="flare", norm=norm, cbar=False)
    axes[2].set_title("Differential Evolution")
    axes[2].set_xlabel("Number of Satellites")
    axes[2].set_ylabel("")

    for ax in axes:
        ax.set_aspect('equal')  # forces square cells

    # Single linear colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    sm = plt.cm.ScalarMappable(cmap='flare', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Data Downlinked in PB over $T_{opt}$")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig("figures_final/heatmap_comparison_vals.pdf", format='pdf', bbox_inches='tight')
    plt.show()

############################# Functions for Prepping Data ###############################

def df_generator(name,csvName,epc_end, epc_start,cfg):

    api = wandb.Api()

    data_downlink_list= []

    # Iterate through the ground station numbers and runs
    try:
        data_downlink_df= pd.read_csv(csvName)
    except FileNotFoundError:
        for gs in [1,2,3,4]:#[1, 2, 3, 4, 5]:
            for sats in [1, 2, 3, 4]:

                project_name = f"loc_gsopt/{name}_{gs}_{sats}=CAPELLA=3000000"
                runs = api.runs(project_name)
                
                for (i,run) in enumerate(runs):
                    
                    if run.state == "finished":
                        gs_list = []
                        for i, new_gs in enumerate(run.summary['gs_list']):
                            point_loc = bh.PointLocation(new_gs[0], new_gs[1])
                            point_loc.set_id(i)
                            gs_list.append(point_loc)
                        satellites = make_walker_constellation(
                                epoch=epc_start,
                                altitude_km=cfg.walker.altitude,
                                eccentricity=cfg.walker.eccentricity,
                                inclination=cfg.walker.inclination,
                                num_planes=sats,
                                sats_per_plane= cfg.walker.sats_perplane,
                                phase=cfg.walker.phase,
                                argp=cfg.walker.argp,
                                norad_start=cfg.walker.norad_start,
                                star =cfg.walker.star,
                            )
                        
                        # Step 1: Propagate all satellites in parallel
                        bh.par_propagate_to(satellites, epc_end)

                        contacts, contact_secs = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                        print(f"trying sat {sats} and gs {gs}")
                        _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                        print(f"finishing sat {sats} and gs {gs}")

                        if csvName != "DE_heatmap.csv":
                            data_downlink_list.append({
                                "gs_number": gs,
                                'sats': sats,
                                'WallClockTime': run.summary['_runtime']/60/60,
                                'contact_num0':run.summary['contact_num0'],
                                'contact_num1':run.summary['contact_num1'],
                                'contact_num2':run.summary['contact_num2'],
                                'contact_num3':run.summary['contact_num3'],
                                'contact_num4':run.summary['contact_num4'],
                                'data_downlink0':run.summary['data_downlink0'],
                                'data_downlink1':run.summary['data_downlink1'],
                                'data_downlink2':run.summary['data_downlink2'],
                                'data_downlink3':run.summary['data_downlink3'],
                                'data_downlink4':run.summary['data_downlink4'],
                                'eval_counter0':run.summary['eval_counter0'],
                                'eval_counter1':run.summary['eval_counter1'],
                                'eval_counter2':run.summary['eval_counter2'],
                                'eval_counter3':run.summary['eval_counter3'],
                                'eval_counter4':run.summary['eval_counter4'],
                                'noExclusion': np.sum(contact_secs)*cfg.scenario.datarate*52/1000/1000/1000/1000/1000,
                                'exclusion': np.sum(contacts_exclusion_secs)*cfg.scenario.datarate*52/1000/1000/1000/1000/1000,
                                # "sat_number": sats,
                            })
                        else:
                            data_downlink_list.append({
                                "gs_number": gs,
                                'sats': sats,
                                'WallClockTime': run.summary['_runtime']/60/60,
                                'eval_counter':run.summary['EvalCount'],
                                'noExclusion': np.sum(contact_secs)*cfg.scenario.datarate*52/1000/1000/1000/1000/1000,
                                'exclusion': np.sum(contacts_exclusion_secs)*cfg.scenario.datarate*52/1000/1000/1000/1000/1000,
                                # "sat_number": sats,
                            })

        # Convert the list of dictionaries into a pandas DataFrame
        data_downlink_df = pd.DataFrame(data_downlink_list)
        data_downlink_df.to_csv(csvName, index=False)  

    return data_downlink_df.groupby(['gs_number', 'sats'], as_index=False).mean(numeric_only=True)

def df_generatorPowell(epc_end, epc_start,cfg):

    api = wandb.Api()

    data_downlink_list_powell = []

    # Iterate through the ground station numbers and runs
    try:
        data_downlink_df_powell = pd.read_csv(f'powell_heatmap.csv')
    except FileNotFoundError:
        for gs in [1,2,3,4]:#[1, 2, 3, 4, 5]:
            gs_list = []
            for sats in [1, 2, 3, 4]:
                project_name = f"loc_gsopt/compute_test1free_powell_data_downlink_{gs}_{sats}=CAPELLA=3000000"
                runs = api.runs(project_name)
                
                for run in runs:
                    
                    if run.state == "finished":
                        # Append each run's data as a dictionary to the list
                        for i, new_gs in enumerate(run.summary['gs_list']):
                            point_loc = bh.PointLocation(new_gs[0], new_gs[1])
                            point_loc.set_id(i)
                            gs_list.append(point_loc)
                        satellites = make_walker_constellation(
                                epoch=epc_start,
                                altitude_km=cfg.walker.altitude,
                                eccentricity=cfg.walker.eccentricity,
                                inclination=cfg.walker.inclination,
                                num_planes=sats,
                                sats_per_plane= cfg.walker.sats_perplane,
                                phase=cfg.walker.phase,
                                argp=cfg.walker.argp,
                                norad_start=cfg.walker.norad_start,
                                star =cfg.walker.star,
                            )
                        
                        # Step 1: Propagate all satellites in parallel
                        bh.par_propagate_to(satellites, epc_end)

                        contacts, contact_secs = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                        _, contacts_exclusion_secs = contactExclusion(contacts,cfg)

                        data_downlink_list_powell.append({
                            "gs_number": gs,
                            'sats': run.config['walker']['num_planes'],
                            'WallClockTime': run.summary['_runtime']/60/60,
                            # "sat_number": sats,
                            "Obj_func": run.history()['Obj_func_value'].dropna().values.min(),
                            "data_downlinked": np.sum(contacts_exclusion_secs)*cfg.scenario.datarate/1000000000
                        })

        # Convert the list of dictionaries into a pandas DataFrame
        data_downlink_df_powell = pd.DataFrame(data_downlink_list_powell)
        data_downlink_df_powell.to_csv(f'powell_heatmap.csv', index=False)  

    return data_downlink_df_powell.groupby(['gs_number', 'sats'], as_index=False).mean(numeric_only=True)



def df_generatorNelder(epc_end, epc_start,cfg):

    api = wandb.Api()

    data_downlink_list_score = []

    # Iterate through the ground station numbers and runs
    try:
        data_downlink_df_score = pd.read_csv(f'nelder_heatmap.csv')
    except FileNotFoundError:
        for gs in [1,2,3,4]:#[1, 2, 3, 4, 5]:
            # for sats in [1, 2, 3, 4, 5]:
            project_name = f"loc_gsopt/compute_test1free_nelder_ccgs_data_downlink_{gs}_5=CAPELLA=3000000"
            runs = api.runs(project_name)
            for run in runs:
                if run.state == "finished":

                    data_downlink_list_score.append({
                        "gs_number": gs,
                        'sats': run.config['walker']['num_planes'],
                        'WallClockTime': run.summary['_runtime']/60/60,
                        # "sat_number": sats,
                        "Obj_func": run.history()['Obj_func_value'].dropna().values.min(),
                        "data_downlinked": run.summary.data_downlink/1000000000
                    })

        # Convert the list of dictionaries into a pandas DataFrame
        data_downlink_df_score = pd.DataFrame(data_downlink_list_score)
        data_downlink_df_score.to_csv(f'nelder_heatmap.csv', index=False)  

    avg_data_downlink_df_score_all = data_downlink_df_score.groupby(['gs_number', 'sats'], as_index=False).mean(numeric_only=True)
    return avg_data_downlink_df_score_all


if __name__ == "__main__":
    main()