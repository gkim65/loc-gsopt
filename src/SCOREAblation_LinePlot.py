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

df_agg_nelder1 = pd.read_csv(f'data/csv_Files/nelder1aggregated_runs.csv')
fig, ax = plt.subplots(figsize=(10, 5))
combined_df = pd.read_csv('data/csv_Files/nelderAblate_wide.csv', index_col=0)



# Columns in combined_df are ordered: gentle-cherry-2, smart-sun-3, ..., revived-water-10
run_labels = {
    0: "Run 1:",
    1: "Run 2:",
    2: "Run 3:",
    3: "Run 4:",
    4: "Run 5:",
    5: "Run 6:",
    6: "Run 7:",
    7: "Run 8:",
    8: "Run 9:",
}
from scipy.ndimage import uniform_filter1d
from matplotlib.patches import Rectangle

data_xlim = (0, 5000)
data_ylim = (0, 4.3)
pad_x = data_xlim[1] * 0.03
pad_y = data_ylim[1] * 0.03

fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)

score_mean = df_agg_nelder1["mean"] + .48
score_steps = df_agg_nelder1["step"]

for i, ax in enumerate(axes.flat):
    col = combined_df.columns[i]
    run_data = (combined_df[col]).values
    # smoothed = uniform_filter1d(run_data, size=50)

    ax.plot(combined_df.index, run_data+ .48, color="#7b52ab", alpha = 1, linewidth=2)
    ax.plot(score_steps, score_mean, color="orange", linewidth=2, linestyle="--")

    ax.axvline(x=800, color="black", linestyle="--", linewidth=2, alpha=0.6, zorder=1)
    # optional small label only on top-left subplot
    if i != 8:
        ax.text(900, 0.2, " Refinement\n begins", fontsize=15, color="black", va="bottom")
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

    ax.set_yticks([0, 1.0, 2.0, 3.0, 4.0])
    ax.set_xticks([0, 1500,3000,4500])
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Title above with enough room — use pad not tight
    ax.set_title(run_labels[i], fontsize=20, pad=6, linespacing=1.4)


fig.text(0.5, 0.01, "Number of Function Evaluations", ha="center", fontsize=22)
fig.text(0.01, 0.5, r"Data Downlinked in $T_{opt}$ (PB)", va="center",
         rotation="vertical", fontsize=22)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#7b52ab", linewidth=1.5, label="SCORE \nRandom"),
    Line2D([0], [0], color="orange", linewidth=1.5, linestyle="--", label="SCORE \nNo Random"),
]
fig.legend(handles=legend_elements, loc="lower right",
           bbox_to_anchor=(0.98, 0.12), fontsize=20)

# Key fix for overlap: more hspace
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
plt.subplots_adjust(hspace=0.55, wspace=0.12)

plt.savefig("figures_final/SCOREsweep_grid.pdf", format="pdf",
            bbox_inches="tight", pad_inches=0.3)
