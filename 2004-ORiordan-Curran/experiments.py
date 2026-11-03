"""Run all four conditions (gradual/sudden x no learning/learning) over many seeds in parallel.
The results ar esaved into file results.npz. 
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"        # one BLAS thread per worker; we parallelise over runs instead
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import experiment as E
from genome import GENOME_LEN
from network import evaluate

LEARN_CYCLES = 50
LEARNING_RATE = 3.0
CONDITIONS = [("gradual", 0), ("gradual", LEARN_CYCLES), ("sudden", 0), ("sudden", LEARN_CYCLES)]
RESULTS_FILE = "results.npz"

def key(mode, cycles, seed):
    """Name of one run inside results.npz; the arrays are stored as <key>_mean and <key>_best."""
    return f"{mode}_c{cycles}_s{seed}"

def job(args):
    mode, cycles, seed, generations = args
    t0 = time.time()
    mean_f, best_f = E.run(mode, cycles, seed=seed, generations=generations, learning_rate=LEARNING_RATE)
    return mode, cycles, seed, mean_f, best_f, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--generations", type=int, default=E.GENERATIONS)
    ap.add_argument("--workers", type=int, default=cpu_count())
    a = ap.parse_args()

    # sanity check: a genome with no nodes outputs a constant 0.5, which must score 0 (decision-based fitness)
    env = E.Environment()
    assert evaluate("0" * GENOME_LEN, env.X, env.t) == 0.0, "network.raw_fitness is not decision-based"

    # Load runs finished earlier (if any) and only queue the missing ones, learning runs first (slowest).
    results = dict(np.load(RESULTS_FILE)) if os.path.exists(RESULTS_FILE) else {}
    jobs = [(m, c, s, a.generations)
            for m, c in sorted(CONDITIONS, key=lambda mc: -mc[1])
            for s in range(a.repeats)
            if key(m, c, s) + "_mean" not in results]
    print(f"{len(jobs)} runs to do on {a.workers} workers", flush=True)
 
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (m, c, s, mean_f, best_f, dt) in enumerate(pool.imap_unordered(job, jobs), 1):
            results[key(m, c, s) + "_mean"] = mean_f
            results[key(m, c, s) + "_best"] = best_f
            np.savez(RESULTS_FILE, **results)          # rewrite the file after every run
            print(f"[{i}/{len(jobs)}] {m} cycles={c} seed={s} done in {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
 