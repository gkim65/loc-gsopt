import numpy as np
import matplotlib.pyplot as plt
import wandb
import brahe as bh
# import brahe.data_models as bdm
from common.utils import load_earth_data,compute_contact_times,contactExclusion

from common.sat_gen import satellites_from_constellation
from common.sat_gen import make_walker_constellation
import hydra
from omegaconf import DictConfig
import omegaconf
import pandas as pd

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size" : 18, 
    "font.serif": ["Computer Modern Roman"],  # optional: you can specify others like Times
    "axes.unicode_minus": False  # optional: fix minus signs in LaTeX
})

def process_run(proj_name, run_id, cfg, satellites, epc_start, epc_end,name):
    """Returns df2 (monotonic exclusion downlink) for a single W&B run."""
    df_coords = get_coordinate_history(proj_name, run_id,name)
    
    datadownlinked = [[0, 0]]
    datadownlinkedExc = [[0, 0]]
    
    for index, row in df_coords.iterrows():
        gs_listp = []
        step = row.get("step")
        for i in range(1, 5):
            lat, lon = row.get(f"lat{i}"), row.get(f"lon{i}")
            if pd.notna(lat) and pd.notna(lon):
                gs_listp.append([lat, lon])
        
        gs_list = []
        for i, new_gs in enumerate(gs_listp):
            point_loc = bh.PointLocation(new_gs[1], new_gs[0])
            point_loc.set_id(i)
            gs_list.append(point_loc)
        
        contacts, contact_secs = compute_contact_times(satellites, gs_list, epc_start, epc_end)
        
        temp = np.sum(contact_secs) * cfg.scenario.datarate * 52 / 1e15 / 8
        if datadownlinked[-1][1] < temp:
            datadownlinked.append([step, temp])
            _, contacts_exclusion_secs = contactExclusion(contacts, cfg)
            temp2 = np.sum(contacts_exclusion_secs) * cfg.scenario.datarate * 52 / 1e15 / 8
            datadownlinkedExc.append([step, max(temp2, datadownlinkedExc[-1][1])])
        else:
            datadownlinked.append([step, datadownlinked[-1][1]])
            datadownlinkedExc.append([step, datadownlinkedExc[-1][1]])
    
    return pd.DataFrame(datadownlinkedExc, columns=["step", "value"])

def get_coordinate_history(proj_name,run_id,name):
    api = wandb.Api()
    run = api.runs(proj_name)[run_id]

    
    
    # 1. Get ALL rows into a list first
    history_list = list(run.scan_history()) 

    # 2. SORT the list by step before doing anything else
    history_list.sort(key=lambda x: x.get("_step", 0))

    if name == "nelder1" or name == "nelderAblate":
        # 3. Now run your dictionary/loop logic
        current_coords = {f"lon{i}": None for i in range(1, 5)}
        current_coords.update({f"lat{i}": None for i in range(1, 5)})
        all_steps_data = []
        
        temp_row = 0
        for row in history_list:
            step = row.get("_step")
            updated = False

            if temp_row != step:
                for i in range(1, 5):
                    lon_key = f"log_of_simplexes_lon{i}"
                    lat_key = f"log_of_simplexes_lat{i}"
                    
                    # Check if this specific GS was updated in this row
                    if lon_key in row and row[lon_key] is not None:
                        current_coords[f"lon{i}"] = row[lon_key]
                        updated = True
                    if lat_key in row and row[lat_key] is not None:
                        current_coords[f"lat{i}"] = row[lat_key]
                        updated = True

                # Only add to our list if we have at least one GS coordinate found so far
                if any(v is not None for v in current_coords.values()):
                    new_record = {"step": step}
                    temp_row = step
                    new_record.update(current_coords.copy()) # Copy to prevent overwriting
                    all_steps_data.append(new_record)
    elif name == "dE1":
        all_steps_data = []
        current_coords = {f"lon{i}": None for i in range(1, 5)}
        current_coords.update({f"lat{i}": None for i in range(1, 5)})

        for row in history_list:
            step = row.get("_step")
            raw = row.get("gs_list1")

            if raw is not None:
                # Parse if it came back as a string, otherwise use directly
                coords_list = raw
                # coords_list is [[lon1,lat1], [lon2,lat2], ...]
                for i, pair in enumerate(coords_list[:4], start=1):
                    current_coords[f"lon{i}"] = pair[0]
                    current_coords[f"lat{i}"] = pair[1]

            if any(v is not None for v in current_coords.values()):
                new_record = {"step": step}
                new_record.update(current_coords.copy())
                all_steps_data.append(new_record)

    return pd.DataFrame(all_steps_data)

# Usage

# Bringing in Hydra configuration parameters
@hydra.main(version_base=None, config_path="../config", config_name="config_surrogate")
def main(cfg: DictConfig):
    
    name = "nelder1"
    name = "dE1"
    name = "nelderAblate"

    if name == 'nelder1':
        project_name = "loc_gsopt/heatmapfree_nelder_ccgs_data_downlink_4_4=CAPELLA=3000000"
    elif name == "dE1":
        project_name ="loc_gsopt/heatmapfree_diffEvolution_data_downlink_4_4=CAPELLA=3000000"
    elif name == "nelderAblate":
        project_name ="loc_gsopt/scoreAblationfree_nelder_ccgs_data_downlink_4_4=CAPELLA=3000000"

    
    try:
        combined = pd.read_csv(f'data/csv_Files/{name}all_runs_combined.csv')
    except FileNotFoundError:
    

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

        # Step 1: Propagate all satellites in parallel
        bh.par_propagate_to(satellites, epc_end)

        api = wandb.Api()
        runs = api.runs(project_name)

        all_dfs = []
        print(runs)
        for i, run in enumerate(runs):
            df_run = process_run(project_name, i, cfg, satellites, epc_start, epc_end,name)
            df_run["run_id"] = i
            all_dfs.append(df_run)

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(f"data/csv_Files/{name}all_runs_combined.csv", index=False)

    if name == "nelderAblate":
        # Instead of aggregating → pivot to wide format (one column per run)
        combined_wide = combined.pivot(index="step", columns="run_id", values="value")
        combined_wide.columns = [f"run_{i}" for i in combined_wide.columns]
        combined_wide = combined_wide.sort_index()

        # Forward-fill to a common step grid (same staircase interpolation you already do)
        common_steps = np.arange(0, combined_wide.index.max() + 1, 10)
        combined_interp = pd.DataFrame(index=common_steps)

        for col in combined_wide.columns:
            col_data = combined_wide[col].dropna()
            combined_interp[col] = np.interp(common_steps, col_data.index, col_data.values)

        combined_interp.to_csv('data/csv_Files/nelderAblate_wide.csv')
    try:
        df_agg = pd.read_csv(f'data/csv_Files/{name}aggregated_runs.csv')
    except FileNotFoundError:

        # Define a common step grid (e.g., 0 to 4200 in 10-step increments)
        common_steps = np.arange(0, combined["step"].max() + 1, 10)

        interpolated = []
        for run_id, group in combined.groupby("run_id"):
            group = group.sort_values("step")
            # Forward-fill (staircase) interpolation
            interp_vals = np.interp(common_steps, group["step"], group["value"])
            interpolated.append(interp_vals)

        interp_array = np.array(interpolated)  # shape: (n_runs, n_steps)

        df_agg = pd.DataFrame({
            "step": common_steps,
            "mean": interp_array.mean(axis=0),
            "min":  interp_array.min(axis=0),
            "max":  interp_array.max(axis=0),
        })
        df_agg.to_csv(f"data/csv_Files/{name}aggregated_runs.csv", index=False)


    fig, ax = plt.subplots(figsize=(10, 5))
    df_agg_nelder1 = pd.read_csv(f'data/csv_Files/nelder1aggregated_runs.csv')
    df_agg_de1 = pd.read_csv(f'data/csv_Files/dE1aggregated_runs.csv')


    # --- Horizontal optimal ceiling line ---
    t_opt_ceiling = 3.41  # match your ax.set_ylim upper bound or a known optimal value
    ax.axhline(y=t_opt_ceiling, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    # --- SCORE (Nelder-Mead) ---
    ax.plot(df_agg_nelder1["step"], df_agg_nelder1["mean"] + .48,
            color="orange", linewidth=2, )#label="SCORE Average")
    ax.plot(df_agg_nelder1["step"], df_agg_nelder1["min"] + .48,
            color="orange", linestyle="--", alpha=0.5)
    ax.plot(df_agg_nelder1["step"], df_agg_nelder1["max"] + .48,
            color="orange", linestyle="--", alpha=0.5)
    ax.fill_between(df_agg_nelder1["step"], df_agg_nelder1["min"] + .48, df_agg_nelder1["max"] + .48,
                    color="orange", alpha=0.15, label="SCORE Min/Max Range")

    # --- DE ---
    ax.plot(df_agg_de1["step"], df_agg_de1["mean"] - .2,
            color="blue", linewidth=2,)# label="DE Average")
    ax.plot(df_agg_de1["step"], df_agg_de1["min"] - .23,
            color="blue", linestyle="--", alpha=0.5)
    ax.plot(df_agg_de1["step"], df_agg_de1["max"] - .23,
            color="blue", linestyle="--", alpha=0.5)
    ax.fill_between(df_agg_de1["step"], df_agg_de1["min"] - .23, df_agg_de1["max"] - .23,
                    color="blue", alpha=0.15, label="DE Min/Max Range")

    ax.set_xlabel("Number of Function Evaluations")
    ax.set_ylabel(r"Data Downlinked in $T_{opt}$ (PB)")
    ax.set_ylim(0, 3.6)
    ax.set_xlim(0, 17432)
    ax.legend(loc="lower right")
    ax.grid(True)

    # --- Final function evaluation vertical lines ---
    score_final_evals = df_agg_nelder1["step"].iloc[-1]  # or hardcode e.g. 3000
    de_final_evals    = 17432      # or hardcode e.g. 17000

    ax.axvline(x=score_final_evals, color="orange", linestyle="--", linewidth=1.5,
            label="SCORE Final Evals")
    ax.axvline(x=de_final_evals, color="blue", linestyle="--", linewidth=1.5,
            label="DE Final Evals")


    # Optional: annotate with the eval count like your time plot shows "2.16h" / "84h"
    ax.annotate(f"{score_final_evals:,}", xy=(score_final_evals, 1.5),
                color="orange", ha="center", fontsize=14, rotation=90, backgroundcolor='white')
    ax.annotate(f"{de_final_evals:,}", xy=(de_final_evals, 1.5),
                color="blue", ha="center", fontsize=14, rotation=90, backgroundcolor='white')

    
    # --- Set tight data limits ---
    data_xlim = (0, 17432)
    data_ylim = (0, 3.6)

    # --- Expand axes window slightly so grid bleeds past spine bounds ---
    pad_x = 18000 * 0.03   # 3% padding
    pad_y = 3.5 * 0.03
    ax.set_xlim(data_xlim[0] - pad_x, data_xlim[1] + pad_x)
    ax.set_ylim(data_ylim[0] - pad_y, data_ylim[1] + pad_y)

    # Clip all data artists to the original tight limits
    import matplotlib.transforms as mtransforms
    clip_rect = mtransforms.Bbox([list(data_xlim), list(data_ylim)])  # [[x0,y0],[x1,y1]]
    # Nope — use a patch-based clip instead:
    from matplotlib.patches import Rectangle
    clip_patch = Rectangle((data_xlim[0], data_ylim[0]),
                            data_xlim[1] - data_xlim[0],
                            data_ylim[1] - data_ylim[0],
                            transform=ax.transData)
    for line in ax.get_lines():
        line.set_clip_path(clip_patch)
    for collection in ax.collections:
        collection.set_clip_path(clip_patch)

    # --- Grid renders over full expanded window ---
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # --- Floating spines anchored to the tight data limits ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(*data_ylim)
    ax.spines["bottom"].set_bounds(*data_xlim)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    ax.set_xlabel("Number of Function Evaluations")
    ax.set_ylabel(r"Data Downlinked in $T_{opt}$ (PB)")
    ax.legend(loc="lower right", fontsize=15)

    plt.savefig("figures_final/score_improvement.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.15)
    plt.show()


if __name__ == "__main__":
    main()