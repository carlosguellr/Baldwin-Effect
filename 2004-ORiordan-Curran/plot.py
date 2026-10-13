"""Plot averaged results (like Figures 1-4 of Curran & O'Riordan) and print summary numbers."""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRADUAL_EVERY, SUDDEN_AT = 19, 150


def load(dir_, mode, cycles):
    files = sorted(glob.glob(os.path.join(dir_, f"{mode}_c{cycles}_s*.npz")))
    if not files:
        return None
    return np.array([np.load(f)["mean"] for f in files])      # (runs, generations)


def change_gens(mode, n_gen):
    if mode == "sudden":
        return [SUDDEN_AT]
    return [g for g in range(GRADUAL_EVERY, n_gen, GRADUAL_EVERY)][:16]


def summarise(mode, runs, n_gen):
    m = runs.mean(axis=0)
    gens = change_gens(mode, n_gen)
    drops = [m[g - 1] - m[g] for g in gens]
    s = f"  runs={len(runs)}  start={m[0]:.2f}  peak={m.max():.2f} (gen {m.argmax()})  end={m[-1]:.2f}"
    if mode == "sudden":
        pre = m[SUDDEN_AT - 10:SUDDEN_AT].mean()
        s += f"\n  before change (gens 140-149)={pre:.2f}  at gen 150={m[SUDDEN_AT]:.2f}  drop={m[SUDDEN_AT - 1] - m[SUDDEN_AT]:.2f}"
        s += f"  gen 160={m[SUDDEN_AT + 10]:.2f}  gen 200={m[SUDDEN_AT + 50]:.2f}"
    else:
        s += f"\n  mean drop per change={np.mean(drops):.3f}  (largest {max(drops):.3f})"
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--learn-cycles", type=int, default=50)
    ap.add_argument("--out", default="figures.png")
    a = ap.parse_args()

    panels = [("gradual", 0, "Gradual change - no lifetime learning"),
              ("gradual", a.learn_cycles, f"Gradual change - lifetime learning ({a.learn_cycles} cycles)"),
              ("sudden", 0, "Sudden change - no lifetime learning"),
              ("sudden", a.learn_cycles, f"Sudden change - lifetime learning ({a.learn_cycles} cycles)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for ax, (mode, cycles, title) in zip(axes.flat, panels):
        runs = load(a.dir, mode, cycles)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Generation")
        ax.grid(alpha=0.3)
        if runs is None:
            ax.text(0.5, 0.5, "no results", ha="center", transform=ax.transAxes)
            continue
        n_gen = runs.shape[1]
        m, sd = runs.mean(axis=0), runs.std(axis=0)
        for g in change_gens(mode, n_gen):
            ax.axvline(g, color="grey", ls=":", lw=0.7)
        ax.fill_between(range(n_gen), m - sd, m + sd, alpha=0.2)
        ax.plot(m, lw=1.3)
        print(f"{title}\n{summarise(mode, runs, n_gen)}")
    axes[0, 0].set_ylabel("Mean fitness (fraction correct)")
    axes[1, 0].set_ylabel("Mean fitness (fraction correct)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("saved", a.out)