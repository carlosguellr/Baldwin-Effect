"""Run all four conditions (gradual/sudden x no learning/learning) over many seeds in parallel.
Each finished run is saved to results/<mode>_c<cycles>_s<seed>.npz, so the script can be
stopped and restarted: finished runs are skipped."""
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
OUT_DIR = "results"


def path(mode, cycles, seed):
    return os.path.join(OUT_DIR, f"{mode}_c{cycles}_s{seed}.npz")


def job(args):
    mode, cycles, seed, generations = args
    t0 = time.time()
    mean_f, best_f = E.run(mode, cycles, seed=seed, generations=generations, learning_rate=LEARNING_RATE)
    np.savez(path(mode, cycles, seed), mean=mean_f, best=best_f)
    return mode, cycles, seed, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--generations", type=int, default=E.GENERATIONS)
    ap.add_argument("--workers", type=int, default=cpu_count())
    a = ap.parse_args()

    # sanity check: a genome with no nodes outputs a constant 0.5, which must score 0 (decision-based fitness)
    env = E.Environment()
    assert evaluate("0" * GENOME_LEN, env.X, env.t) == 0.0, "network.raw_fitness is not decision-based"

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [(m, c, s, a.generations) for c in sorted({c for _, c in CONDITIONS}, reverse=True)
            for m, cc in CONDITIONS if cc == c for s in range(a.repeats)
            if not os.path.exists(path(m, c, s))]
    print(f"{len(jobs)} runs to do on {a.workers} workers (learning runs are queued first)", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (m, c, s, dt) in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {m} cycles={c} seed={s} done in {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)