"""
Reads the per-seed CSVs written by run_experiment.py, averages them over
seeds, and plots the result

produces
    <condition>/summary.csv   per-generation mean and standard deviation over seeds
    comparison.png            mean test MAE of the best genome per generation for every
                              condition, with a +/- 1 std band, plus the innate
                              (pre-learning) test MAE for every condition that learns.
                              If it already exists, saves as comparison_1.png, _2, ... instead

Usage
    python plot_results.py

Every condition folder found in results/ is included automatically.
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

from friedman_data import DRIFT_GENS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Colours and ink (validated categorical palette, light mode), one fixed colour per condition
COLOURS = {"darwinian": "#2a78d6", "baldwinian": "#eb6834", "lamarckian": "#1baf7a"}
ORDER = ["darwinian", "baldwinian", "lamarckian"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

METRICS = ["val_MAE", "test_MAE", "innate_test_MAE", "pop_mean_val_MAE",
           "hidden_nodes", "enabled_connections", "n_params"]


def load_runs(condition):
    """Load every seed CSV for a condition into one long DataFrame."""
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, condition, "seed_*.csv")))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def summarise(runs):
    """Mean and standard deviation over seeds, for every generation."""
    summary = runs.groupby("generation")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]   # flatten column names
    return summary.reset_index()


def unique_path(path):
    """Return path unchanged if it is free, otherwise add _1, _2, ... before the extension."""
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(path):
        path = f"{base}_{i}{ext}"
        i += 1
    return path


def find_conditions():
    """Condition folders in results/ that contain at least one seed CSV, in a fixed order."""
    found = [c for c in ORDER if glob.glob(os.path.join(RESULTS_DIR, c, "seed_*.csv"))]
    if not found:
        raise FileNotFoundError("No seed CSVs in results/ - run run_experiment.py first")
    return found


def plot_mae(summaries, n_seeds, out_path):
    """Line chart of mean best-genome test MAE over generations, one line per condition, with drift markers."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    top = 0
    for cond, s in summaries.items():
        c, g = COLOURS[cond], s["generation"]
        mean, std = s["test_MAE_mean"], s["test_MAE_std"]

        # Test MAE mean (after learning, if the condition learns) with +/- 1 std across seeds
        ax.fill_between(g, mean - std, mean + std, color=c, alpha=0.15, lw=0)
        ax.plot(g, mean, color=c, lw=2, label=f"{cond.capitalize()}: test MAE (n={n_seeds[cond]})")

        # Innate test MAE (birth weights, before learning); same as test MAE for darwinian, so skip it
        if cond != "darwinian":
            ax.plot(g, s["innate_test_MAE_mean"], color=c, lw=1.2, ls="--",
                    label=f"{cond.capitalize()}: innate test MAE (before learning)")
        top = max(top, (mean + std).iloc[3:].max(), s["innate_test_MAE_mean"].iloc[3:].max())

    # y-limit ignores the first few generations, where the random initial population is very poor
    ax.set_ylim(0, top * 1.15)

    for drift_g, txt in zip(DRIFT_GENS, ["drift → concept B", "concept A returns"]):
        ax.axvline(drift_g, color=INK_2, lw=1)
        ax.text(drift_g + 2, ax.get_ylim()[1] * 0.97, txt, color=INK_2, fontsize=9, va="top")

    ax.set_xlabel("Generation", color=INK_2)
    ax.set_ylabel("MAE of best genome", color=INK_2)
    ax.set_title("NEAT on FriedmanDrift: mean over seeds (band = ± 1 std)",
                 loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2)
    # Solid legend background so  drift lines don't run through the text
    leg = ax.legend(fontsize=8.5, loc="lower left", ncol=2, frameon=True, framealpha=1)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    summaries, n_seeds = {}, {}
    for cond in find_conditions():
        runs = load_runs(cond)
        n_seeds[cond] = runs["seed"].nunique()
        summaries[cond] = summarise(runs)

        out_dir = os.path.join(RESULTS_DIR, cond)
        summaries[cond].to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        print(f"{cond}: {n_seeds[cond]} seeds -> {out_dir}/summary.csv")

    out_path = unique_path(os.path.join(RESULTS_DIR, "comparison.png"))
    plot_mae(summaries, n_seeds, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()