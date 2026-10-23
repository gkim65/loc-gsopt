"""
Export the MILP (teleport selection) and KSAT baseline results from W&B to CSV.

The downlink comparison figures in linePlots.py read these baselines from W&B run
summaries. This script writes them to data/csv_Files/ so the paper's numbers are
kept alongside the code and can be read without W&B access.

Run from the repository root after `wandb login`:

    python src/export_baselines.py [--entity ENTITY]

Writes `milp_{CONSTELLATION}.csv` and `ksat_{CONSTELLATION}.csv` with columns
`constellation, gs_number, data_downlinked, run_id, run_name, project`, keeping only
the rows the paper figures plot (see `figure_rows`).
"""

import argparse
import os

import pandas as pd
import wandb

OUT_DIR = "data/csv_Files"

GS_NUMBERS = [1, 2, 3, 4, 5, 7, 10, 15, 20]

# Constellation name -> "<sat_num>=<constellation>" suffix used in the W&B project names
CONSTELLATIONS = {"CAPELLA": "5=CAPELLA", "ICEYE": "34=ICEYE"}

# Baseline name -> (W&B project name template, run summary key holding total data downlinked)
BASELINES = {
    "milp": ("teleport_select_100_data_downlink_{gs}_{suffix}=3000000", "output_data"),
    "ksat": ("KSAT_MILP_data_downlink_{gs}_{suffix}=3000000", "data_downlinked.total"),
}

# (baseline, constellation) -> ground station counts the paper figures leave out
FIGURE_EXCLUDES = {("milp", "ICEYE"): [4]}


def export_baseline(api, entity, baseline, constellation):
    """
        export_baseline(api, entity, baseline, constellation) -> pandas.DataFrame

    Collect one baseline's finished runs for one constellation across all ground station counts.

      - `api` — an authenticated `wandb.Api`
      - `entity` — W&B entity that owns the projects
      - `baseline` — a key of `BASELINES`
      - `constellation` — a key of `CONSTELLATIONS`

    Returns one row per finished run. `gs_number` is the number of ground stations
    selected. `data_downlinked` is the run summary value divided by 1e9, the same
    scaling linePlots.py applies.

    NOTE: projects that do not exist are skipped with a printed message, matching linePlots.py.
    """
    template, summary_key = BASELINES[baseline]
    rows = []
    for gs in GS_NUMBERS:
        project = f"{entity}/" + template.format(gs=gs, suffix=CONSTELLATIONS[constellation])
        try:
            for run in api.runs(project):
                if run.state != "finished":
                    continue
                rows.append({
                    "constellation": constellation,
                    "gs_number": gs,
                    "data_downlinked": run.summary[summary_key] / 1e9,
                    "run_id": run.id,
                    "run_name": run.name,
                    "project": project,
                })
        except Exception as e:
            print(f"Skipping {project}: {e}")
    return pd.DataFrame(rows)


def figure_rows(df, baseline, constellation):
    """
        figure_rows(df, baseline, constellation) -> pandas.DataFrame

    Reduce exported runs to the rows the paper figures plot: the best run at each
    ground station count, minus the counts in `FIGURE_EXCLUDES`.

      - `df` — output of `export_baseline`
      - `baseline` — a key of `BASELINES`
      - `constellation` — a key of `CONSTELLATIONS`

    Returns one row per plotted ground station count, sorted by `gs_number`.
    """
    df = df[~df["gs_number"].isin(FIGURE_EXCLUDES.get((baseline, constellation), []))]
    best = df.loc[df.groupby("gs_number")["data_downlinked"].idxmax()]
    return best.sort_values("gs_number").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "loc_gsopt"),
                        help="W&B entity that owns the baseline projects")
    args = parser.parse_args()

    api = wandb.Api()
    os.makedirs(OUT_DIR, exist_ok=True)
    for baseline in BASELINES:
        for constellation in CONSTELLATIONS:
            df = figure_rows(export_baseline(api, args.entity, baseline, constellation),
                             baseline, constellation)
            path = os.path.join(OUT_DIR, f"{baseline}_{constellation}.csv")
            df.to_csv(path, index=False)
            print(f"Wrote {len(df)} runs to {path}")


if __name__ == "__main__":
    main()
