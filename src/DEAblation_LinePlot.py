import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

plt.figure(figsize=(16, 8))

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size" : 24, 
    "font.serif": ["Computer Modern Roman"],  # optional: you can specify others like Times
    "axes.unicode_minus": False  # optional: fix minus signs in LaTeX
})

def readData_individual(project_name):
    api = wandb.Api()
    runs = api.runs(project_name)

    all_runs_data = []
    hard_floor = -100677433.83433

    for run in runs:
        if True: #run.state == "finished":
            history = list(run.scan_history(keys=["_step", "Obj_func_value"]))
            if not history: continue
            history.sort(key=lambda x: x["_step"])

            baseline = history[0]["Obj_func_value"]
            current_best_improvement = 0.0
            run_steps, run_values = [], []

            for i, row in enumerate(history):
                if i <20000:
                    val = row["Obj_func_value"]
                    diff = val - baseline
                    if diff < current_best_improvement:
                        current_best_improvement = max(diff, hard_floor)
                    run_steps.append(row["_step"])
                    run_values.append(current_best_improvement)
                else:
                    break

            run_df = pd.Series(run_values, index=run_steps, name=run.name)  # use run.name for readable labels
            run_df = run_df[~run_df.index.duplicated(keep='last')]
            all_runs_data.append(run_df)
            print(run_df)

    scale = -1 * 1200000000 * 52 / (1000**5) / 8
    combined_df = pd.concat(all_runs_data, axis=1).sort_index()* scale

    return combined_df  # each column = one run

try:
    combined_df = pd.read_csv(f'data/csv_Files/DEimprovement_plot_data.csv', index_col=0)
except FileNotFoundError:
    project_name = f"loc_gsopt/DE4_4free_diffEvolution_data_downlink_4_4=CAPELLA=3000000"
    combined_df = readData_individual(project_name)
    combined_df.to_csv('data/csv_Files/DEimprovement_plot_data.csv')
    print("Done! CSV ready for plotting.")
    

fig, ax = plt.subplots(figsize=(10, 5))
# combined_df = combined_df   # apply once
print(combined_df.head())
for run_name, series in combined_df.items():
    ax.plot(series.index, series.values/20, linewidth=1.2, alpha=0.7, label=run_name)

ax.set_xlabel("Step")
ax.set_ylabel(r"Data Downlinked in $T_{opt}$ (PB)")
ax.legend(loc="lower right", fontsize=7)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures_final/individual_runs.pdf", format="pdf", bbox_inches="tight", pad_inches=0.15)
# plt.show()

df_agg_nelder1 = pd.read_csv(f'data/csv_Files/nelder1aggregated_runs.csv')
fig, ax = plt.subplots(figsize=(10, 5))



# Columns in combined_df are ordered: gentle-cherry-2, smart-sun-3, ..., revived-water-10
run_labels = {
    0: "Run 1: NP=5d, \n F=0.5, CR=0.3, best1bin",
    1: "Run 2: NP=10d, \n F=0.7, CR=0.9, rand1bin",
    2: "Run 3: NP=10d, \n F=0.9, CR=0.3, best1bin",
    3: "Run 4: NP=15d, \n F=0.7, CR=0.3, rand1bin",
    4: "Run 5: NP=5d, \n F=0.9, CR=0.9, rand1bin",
    5: "Run 6: NP=10d, \n F=0.5, CR=0.9, rand1bin",
    6: "Run 7: NP=15d, \n F=0.9, CR=0.9, rand1bin",
    7: "Run 8: NP=10d, \n F=0.7, CR=0.5, best1bin",
    8: "Run 9: NP=10d, \n F=0.6, CR=0.6, best1bin",
}
from scipy.ndimage import uniform_filter1d
from matplotlib.patches import Rectangle

data_xlim = (0, 17400)
data_ylim = (0, 3.5)
pad_x = data_xlim[1] * 0.03
pad_y = data_ylim[1] * 0.03

fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)

score_mean = df_agg_nelder1["mean"] + .48
score_steps = df_agg_nelder1["step"]

for i, ax in enumerate(axes.flat):
    col = combined_df.columns[i]
    run_data = (combined_df[col] / 20 + .05).values
    smoothed = uniform_filter1d(run_data, size=50)

    ax.plot(combined_df.index, smoothed, color="blue", alpha = 0.7, linewidth=2)
    ax.plot(score_steps, score_mean, color="orange", linewidth=2, linestyle="--")

    # --- Floating spine whitespace style ---
    ax.set_xlim(data_xlim[0] - pad_x, data_xlim[1] + pad_x)
    ax.set_ylim(data_ylim[0] - pad_y, data_ylim[1] + pad_y)

    clip_patch = Rectangle((data_xlim[0], data_ylim[0]),
                            data_xlim[1] - data_xlim[0],
                            data_ylim[1] - data_ylim[0],
                            transform=ax.transData)
    for line in ax.get_lines():
        line.set_clip_path(clip_patch)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(*data_ylim)
    ax.spines["bottom"].set_bounds(*data_xlim)
    ax.spines["left"].set_position(("outward", 8))
    ax.spines["bottom"].set_position(("outward", 8))

    ax.set_yticks([0, 1.0, 2.0, 3.0, 3.5])
    ax.set_xticks([0, 5000, 10000, 15000])
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Title above with enough room — use pad not tight
    ax.set_title(run_labels[i], fontsize=20, pad=6, linespacing=1.4)

fig.text(0.5, 0.01, "Number of Function Evaluations", ha="center", fontsize=22)
fig.text(0.01, 0.5, r"Data Downlinked in $T_{opt}$ (PB)", va="center",
         rotation="vertical", fontsize=22)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="blue", linewidth=1.5, label="DE Run"),
    Line2D([0], [0], color="orange", linewidth=1.5, linestyle="--", label="SCORE Avg"),
]
fig.legend(handles=legend_elements, loc="lower right",
           bbox_to_anchor=(0.98, 0.15), fontsize=24)

# Key fix for overlap: more hspace
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.subplots_adjust(hspace=0.55, wspace=0.12)

plt.savefig("figures_final/DEsweep_grid.pdf", format="pdf",
            bbox_inches="tight", pad_inches=0.3)

# # --- SCORE (Nelder-Mead) ---
# ax.plot(df_agg_nelder1["step"], df_agg_nelder1["mean"] + .48,
#         color="orange", linewidth=2, label="SCORE Average")
# ax.plot(df_agg_nelder1["step"], df_agg_nelder1["min"] + .48,
#         color="orange", linestyle="--", alpha=0.5)
# ax.plot(df_agg_nelder1["step"], df_agg_nelder1["max"] + .48,
#         color="orange", linestyle="--", alpha=0.5)
# ax.fill_between(df_agg_nelder1["step"], df_agg_nelder1["min"] + .48, df_agg_nelder1["max"] + .48,
#                 color="orange", alpha=0.15, label="SCORE Min/Max Range")

# # --- DE individual runs (bottom layer) ---
# de_all = combined_df / 20 + .05

# for col in de_all.columns:
#     ax.plot(de_all.index, de_all[col], linewidth=0.7, alpha=0.12, color="blue")

# # --- DE aggregated band ---
# de_min  = de_all.min(axis=1)
# de_max  = de_all.max(axis=1)
# de_mean = de_all.mean(axis=1)

# ax.fill_between(de_all.index, de_min, de_max,
#                 color="blue", alpha=0.15, label="DE Min/Max Range")
# ax.plot(de_all.index, de_min, color="blue", linestyle="--", alpha=0.5)
# ax.plot(de_all.index, de_max, color="blue", linestyle="--", alpha=0.5)
# ax.plot(de_all.index, de_mean, color="blue", linewidth=2, label="DE Average")

# ax.plot([], [], color="blue", linewidth=1.0, alpha=0.5, label="DE Runs")

# ax.set_xlabel("Number of Function Evaluations")
# ax.set_ylabel(r"Data Downlinked in $T_{opt}$ (PB)")
# ax.set_ylim(0, 3.5)
# ax.set_xlim(0, 17400)
# ax.legend(loc="lower right")
# ax.grid(True, linestyle="--", alpha=0.4)
# plt.tight_layout()
# # ax.set_xlim(0, 17000)
# # ax.set_xlabel("Number of Function Evaluations")
# # ax.set_ylabel(r"Data Downlinked in $T_{opt}$ (PB)")
# # ax.legend(loc="lower right")
# # ax.grid(True, linestyle="--", alpha=0.5)
# # plt.tight_layout()
# plt.savefig("figures_final/score_de_comparison.pdf", format="pdf", bbox_inches="tight", pad_inches=0.15)
# plt.show()