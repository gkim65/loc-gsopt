"""
Export the surrogate-window convergence runs from the local W&B cache to a compressed CSV.

surrogateOptPlot.py caches the run histories of the W&B project in `CACHE_FILE`. This
script keeps only the history rows the convergence figure depends on and writes them to
data/csv_Files/surrogate_convergence.csv.gz.

Run from the repository root:

    python src/export_surrogate.py

Writes columns `run_id, num_days, total_contact_num, total_seconds` (days, contacts,
seconds of contact), cumulative from the start of each run.
"""

import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from surrogateOptPlot import CACHE_FILE, MAX_DAYS

OUT_PATH = "data/csv_Files/surrogate_convergence.csv.gz"


def figure_rows(history):
    """
        figure_rows(history) -> pandas.DataFrame

    Keep the rows of one run's history that the convergence figure reads: every day up to
    `MAX_DAYS`, plus the last row, which sets the run's full-length reference mean.

      - `history` — one run's cumulative history, columns `num_days, total_contact_num, total_seconds`

    Returns the kept rows in day order.

    NOTE: this is exact only because the figure's running mean stops at `MAX_DAYS` and its
    reference mean depends only on the first and last cumulative values.
    """
    keep = (history["num_days"] <= MAX_DAYS) | (history.index == history.index[-1])
    return history[keep]


def main():
    with open(CACHE_FILE, "rb") as f:
        all_data = pickle.load(f)

    rows = []
    for run_id, d in all_data.items():
        history = pd.DataFrame({key: d[key] for key in ["num_days", "total_contact_num", "total_seconds"]})
        history = history.sort_values("num_days").reset_index(drop=True)
        rows.append(figure_rows(history).assign(run_id=run_id))

    df = pd.concat(rows, ignore_index=True)[["run_id", "num_days", "total_contact_num", "total_seconds"]]
    df.to_csv(OUT_PATH, index=False, compression="gzip")
    print(f"Wrote {len(df)} rows from {len(all_data)} runs to {OUT_PATH}")


if __name__ == "__main__":
    main()
