"""Plot averaged results (like Figures 1-4 of Curran & O'Riordan) and print summary numbers."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Draw to file, not to screen
import matplotlib.pyplot as plt

# Generations at which the environment changes. These must match experiment.py
GRADUAL_CYCLES, SUDDEN = 19, 150


def load(data, mode, cycles):
    """Collect every run of one condition into an array of (runs, generations).
    data is the opened results.npz; keys look like 'sudden_c50_s3_mean'.

    Returns None if the file has no runs for that condition."""
    keys = sorted(k for k in data.files if k.startswith(f"{mode}_c{cycles}_s") and k.endswith("_mean"))
    if not keys:
        return None
    return np.array([data[k] for k in keys])


def change_gens(mode, n_gen):
    """Generations at which the environment changes (used for the dotted lines and the drop sizes)."""
    if mode == "sudden":
        return [SUDDEN]
    return [g for g in range(GRADUAL_CYCLES, n_gen, GRADUAL_CYCLES)][:16]


def summarise(mode, runs, n_gen):
    """Text summary of the averaged curve: start, peak, end, and the size of the fitness drop(s)."""
    m = runs.mean(axis=0)                                     # average over runs, per generation
    gens = change_gens(mode, n_gen)
    drops = [m[g - 1] - m[g] for g in gens]                   # fitness lost at each change
    s = f"  runs={len(runs)}  start={m[0]:.2f}  peak={m.max():.2f} (gen {m.argmax()})  end={m[-1]:.2f}"
    if mode == "sudden":
        pre = m[SUDDEN - 10:SUDDEN].mean()              # level just before the change
        s += f"\n  before change (gens 140-149)={pre:.2f}  at gen 150={m[SUDDEN]:.2f}  drop={m[SUDDEN - 1] - m[SUDDEN]:.2f}"
        s += f"  gen 160={m[SUDDEN + 10]:.2f}  gen 200={m[SUDDEN + 50]:.2f}"   # recovery
    else:
        s += f"\n  mean drop per change={np.mean(drops):.3f}  (largest {max(drops):.3f})"
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results.npz", help="results file written by run_experiments.py")
    ap.add_argument("--learn-cycles", type=int, default=50, help="cycle count used in the learning runs")
    ap.add_argument("--out", default="figures.png", help="output image")
    a = ap.parse_args()
 
    data = np.load(a.file)                                    # open the results file once
 
    # The four panels, in the order of the paper's Figures 1-4: (mode, training cycles, title).
    panels = [("gradual", 0, "Gradual change - no lifetime learning"),
              ("gradual", a.learn_cycles, f"Gradual change - lifetime learning ({a.learn_cycles} cycles)"),
              ("sudden", 0, "Sudden change - no lifetime learning"),
              ("sudden", a.learn_cycles, f"Sudden change - lifetime learning ({a.learn_cycles} cycles)")]
 
    # 2x2 grid: top row = gradual, bottom row = sudden; the two panels of a row share the y-axis
    # so that the effect of learning can be compared directly.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for ax, (mode, cycles, title) in zip(axes.flat, panels):
        runs = load(data, mode, cycles)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Generation")
        ax.grid(alpha=0.3)
        if runs is None:                                      # condition not run (yet)
            ax.text(0.5, 0.5, "no results", ha="center", transform=ax.transAxes)
            continue
 
        n_gen = runs.shape[1]
        mean_curve = runs.mean(axis=0)                        # average over the repeated runs
        for g in change_gens(mode, n_gen):                    # dotted line at every environment change
            ax.axvline(g, color="grey", ls=":", lw=0.7)
        ax.plot(mean_curve, lw=1.3)                           # the averaged fitness curve
        print(f"{title}\n{summarise(mode, runs, n_gen)}")
 
    axes[0, 0].set_ylabel("Mean fitness (fraction correct)")
    axes[1, 0].set_ylabel("Mean fitness (fraction correct)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("saved", a.out)
 