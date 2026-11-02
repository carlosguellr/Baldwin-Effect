"""
Builds the drifting regression environment used by every experiment.

river's ``FriedmanDrift`` generator with ``drift_type="gra"``(Global Recurring Abrupt)

The inputs are always 10 features drawn uniformly from [0, 1]; 
only the formula that maps inputs to the target changes:
    concept A:  y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 + noise
    concept B:  y = 10 sin(pi x3 x5) + 20 (x1 - 0.5)^2 + 10 x0 + 5 x2 + noise

    noise ~ N(0, 1)  ->  best achievable MAE is about 0.80

The stream is cut into three blocks: A, then B, then fresh samples of A.
Each block is split once into train / validation / test:

    train (2000) -> lifetime learning (only used by Baldwinian / Lamarckian NEAT)
    val   (500)  -> fitness, what selection sees
    test  (1000) -> reporting only, never seen by evolution
"""
import numpy as np
from river.datasets import synth

GENERATIONS = 300            
DRIFT_GENS = (100, 200)      # generation at which concept B starts, and A returns
CONCEPTS = "ABA"             # concept label for each block

N_TRAIN, N_VAL, N_TEST = 2000, 500, 1000
BLOCK = N_TRAIN + N_VAL + N_TEST


# Friedman targets lie roughly in 0..30. Networks work on roughly -1..1,
# so network output o is mapped back to the target scale as o * Y_SCALE + Y_CENTRE.
Y_CENTRE, Y_SCALE = 15.0, 15.0


def make_blocks(seed):
    """Generate the three concept blocks (A, B, A) and split each one.

    Returns
    list of dict
        One dict per block with keys "train", "val", "test".
        Each value is a tuple (X, y) with X of shape (n, 10) and y of shape (n,).
    """
    # The generator switches formula at sample BLOCK and back at 2 * BLOCK
    stream = synth.FriedmanDrift(drift_type="gra", position=(BLOCK, 2 * BLOCK), seed=seed)

    X, y = [], []
    for x_dict, target in stream.take(3 * BLOCK):
        X.append([x_dict[i] for i in range(10)])   # river yields dicts; convert to a row
        y.append(target)
    X, y = np.array(X), np.array(y)

    blocks = []
    for b in range(3):
        Xb = X[b * BLOCK:(b + 1) * BLOCK]
        yb = y[b * BLOCK:(b + 1) * BLOCK]
        blocks.append({
            "train": (Xb[:N_TRAIN], yb[:N_TRAIN]),
            "val": (Xb[N_TRAIN:N_TRAIN + N_VAL], yb[N_TRAIN:N_TRAIN + N_VAL]),
            "test": (Xb[N_TRAIN + N_VAL:], yb[N_TRAIN + N_VAL:]),
        })
    return blocks


def block_for_generation(g):
    """Return which concept block (0, 1 or 2) is active at generation g."""
    if g < DRIFT_GENS[0]:
        return 0
    return 1 if g < DRIFT_GENS[1] else 2