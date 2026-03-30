import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.figure(figsize=(16, 8))

mpl.rcParams.update({
    "text.usetex": False,
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
plt.show()


df_agg_nelder1 = pd.read_csv(f'data/csv_Files/nelder1aggregated_runs.csv')
fig, ax = plt.subplots(figsize=(10, 5))

# --- SCORE (Nelder-Mead) ---
ax.plot(df_agg_nelder1["step"], df_agg_nelder1["mean"] + .4,
        color="orange", linewidth=2, label="SCORE Average")
ax.plot(df_agg_nelder1["step"], df_agg_nelder1["min"] + .4,
        color="orange", linestyle="--", alpha=0.5)
ax.plot(df_agg_nelder1["step"], df_agg_nelder1["max"] + .4,
        color="orange", linestyle="--", alpha=0.5)
ax.fill_between(df_agg_nelder1["step"], df_agg_nelder1["min"] + .4, df_agg_nelder1["max"] + .4,
                color="orange", alpha=0.15, label="SCORE Min/Max Range")

# --- DE individual runs ---
for run_name, series in combined_df.items():
    ax.plot(series.index, series.values/20, linewidth=1.0, alpha=0.6, color="blue")
ax.plot([], [], color="blue", linewidth=1.0, label="DE Runs")  # legend proxy

ax.set_xlim(0, 17000)
ax.set_xlabel("Number of Function Evaluations")
ax.set_ylabel(r"Data Downlinked in $T_{opt}$ (PB)")
ax.legend(loc="lower right")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures_final/score_de_comparison.pdf", format="pdf", bbox_inches="tight", pad_inches=0.15)
plt.show()