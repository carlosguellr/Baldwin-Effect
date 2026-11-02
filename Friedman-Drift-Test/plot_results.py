"""
Reads the per-seed CSVs written by run_experiment.py, averages them over
seeds, and plots the result

What it produces (in results/<condition>/)
-----------------------------------------
summary.csv   per-generation mean and standard deviation over seeds
mae.png       mean MAE of the best genome per generation, with a +/- 1 std band

Usage
-----
    python plot_results.py                       # condition "darwinian"
    python plot_results.py --condition darwinian
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

from friedman_data import DRIFT_GENS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Colours and ink (validated categorical palette, light mode)
C_TEST, C_VAL = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(condition):
    """Load every seed CSV for a condition into one long DataFrame."""
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, condition, "seed_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No seed CSVs in results/{condition}/ - run run_experiment.py first")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def summarise(runs):
    """Mean and standard deviation over seeds, for every generation."""
    metrics = ["val_MAE", "test_MAE", "innate_test_MAE", "hidden_nodes", "enabled_connections"]
    summary = runs.groupby("generation")[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]   # flatten column names
    return summary.reset_index()


def plot_mae(summary, n_seeds, condition, out_path):
    """Line chart of mean best-genome MAE over generations, with drift markers."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    g = summary["generation"]

    # Validation MAE mean
    ax.plot(g, summary["val_MAE_mean"], color=C_VAL, lw=1.5, label="Validation MAE (fitness)")

    # Test MAE mean with +/- 1 std across seeds
    ax.fill_between(g, summary["test_MAE_mean"] - summary["test_MAE_std"],
                    summary["test_MAE_mean"] + summary["test_MAE_std"],
                    color=C_TEST, alpha=0.15, lw=0, label="Test MAE ± 1 std")
    ax.plot(g, summary["test_MAE_mean"], color=C_TEST, lw=2, label="Test MAE (held out)")

    # y-limit ignores the first few generations, where the random initial population is very poor
    top = (summary["test_MAE_mean"] + summary["test_MAE_std"]).iloc[3:].max()
    ax.set_ylim(0, top * 1.2)

    for drift_g, txt in zip(DRIFT_GENS, ["drift → concept B", "concept A returns"]):
        ax.axvline(drift_g, color=INK_2, lw=1)
        ax.text(drift_g + 2, ax.get_ylim()[1] * 0.97, txt, color=INK_2, fontsize=9, va="top")

    ax.set_xlabel("Generation", color=INK_2)
    ax.set_ylabel("MAE of best genome", color=INK_2)
    ax.set_title(f"{condition.capitalize()} NEAT on FriedmanDrift, mean of {n_seeds} runs",
                 loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2)
    # Solid legend background so  drift lines don't run through the text
    leg = ax.legend(fontsize=9, loc="lower left", ncol=2, frameon=True, framealpha=1)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Average and plot NEAT drift results.")
    parser.add_argument("--condition", default="darwinian")
    args = parser.parse_args()

    runs = load_runs(args.condition)
    n_seeds = runs["seed"].nunique()
    summary = summarise(runs)

    out_dir = os.path.join(RESULTS_DIR, args.condition)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    plot_mae(summary, n_seeds, args.condition, os.path.join(out_dir, "mae.png"))
    print(f"{args.condition}: {n_seeds} seeds -> {out_dir}/summary.csv, {out_dir}/mae.png")


if __name__ == "__main__":
    main()