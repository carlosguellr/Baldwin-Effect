"""
Runs NEAT on the drifting Friedman environment and saves one CSV per seed.

darwinian : weights are fixed at birth, no lifetime learning.
            Each genome is scored as-is on the validation set.

baldwinian : same as darwinia, but lifetime learning is performed on the training set before scoring on the validation set.

lamackrian : same as Baldwinian, but the learned weights are inherited by the offspring.


Per generation, each run logs the BEST genome of the population:
    val_MAE              validation MAE (what selection used, i.e. -fitness)
    test_MAE             MAE on the held-out test set of the current concept
    innate_test_MAE      test MAE before any lifetime learning
                         (identical to test_MAE for darwinian)
    hidden_nodes         number of hidden nodes in the best genome
    enabled_connections  number of active connections in the best genome
    n_species            number of species in the population


outputs results in results/<condition>/seed_<NN>.csv
"""
import os
import random
import time
from multiprocessing import Pool
 
import neat
import numpy as np
import pandas as pd
 
from friedman_data import (CONCEPTS, GENERATIONS, Y_CENTRE, Y_SCALE,
                           block_for_generation, make_blocks)
 
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "neat_config.ini")
RESULTS_DIR = os.path.join(HERE, "results")
 
N_SEEDS = 20               
WORKERS = os.cpu_count()   # parallel processes; set to 1 to run seeds one after another
 
 
# ---------- evaluation ----------
def mae(net, data):
    """Mean absolute error of a NEAT network on (X, y)"""
    X, y = data
    pred = np.array([net.activate(row)[0] for row in X]) * Y_SCALE + Y_CENTRE
    return float(np.mean(np.abs(pred - y)))
 
 
def make_eval_function(state):
    """Build the fitness function neat-python calls once per generation.
 
    ``state["block"]`` holds the concept block. It is updated by the
    run loop when the environment drifts, so the same function always scores
    genomes on the current concept.
    """
    def eval_genomes(genomes, config):
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # Selection uses the validation set
            genome.fitness = -mae(net, state["block"]["val"])
    return eval_genomes
 
 
def run_seed(seed, condition="darwinian", verbose=False):
    """Run one full evolutionary run and save its per-generation log."""
    random.seed(seed)          # neat-python uses Python's random module
    np.random.seed(seed)
    blocks = make_blocks(seed)
 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)
    pop = neat.Population(config)
 
    state = {"block": None}
    eval_genomes = make_eval_function(state)
 
    log, t0 = [], time.time()
    for g in range(GENERATIONS):
        b = block_for_generation(g)
        state["block"] = blocks[b]                 # switch concept at the drift points
 
        best = pop.run(eval_genomes, 1)            # evolve exactly one generation
        net = neat.nn.FeedForwardNetwork.create(best, config)
        test_mae = mae(net, blocks[b]["test"])
 
        log.append({
            "seed": seed,
            "generation": g,
            "concept": CONCEPTS[b],
            "val_MAE": -best.fitness,
            "test_MAE": test_mae,
            "innate_test_MAE": test_mae,           # no lifetime learning in darwinian
            "hidden_nodes": len(best.nodes) - config.genome_config.num_outputs,
            "enabled_connections": sum(c.enabled for c in best.connections.values()),
            "n_species": len(pop.species.species),
        })
        if verbose and (g % 25 == 0 or g == GENERATIONS - 1):
            r = log[-1]
            print(f"[seed {seed}] gen {g:3d} [{r['concept']}]  val {r['val_MAE']:.2f}  "
                  f"test {r['test_MAE']:.2f}  hidden {r['hidden_nodes']}  "
                  f"({time.time() - t0:.0f}s)")
 
    out_dir = os.path.join(RESULTS_DIR, condition)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed_{seed:02d}.csv")
    pd.DataFrame(log).to_csv(path, index=False)
    print(f"[seed {seed}] done in {time.time() - t0:.0f}s -> {path}")
    return path
 
# ---------- many runs ----------
def main():
    """Run every seed, in parallel unless WORKERS is 1."""
    seeds = list(range(N_SEEDS))
    t0 = time.time()
    if WORKERS == 1:
        for s in seeds:
            run_seed(s, verbose=True)
    else:
        # Each seed is independent, so seeds run in parallel processes.
        with Pool(WORKERS) as pool:
            pool.map(run_seed, seeds)
    print(f"All {len(seeds)} seeds finished in {(time.time() - t0) / 60:.1f} min")
 
 
if __name__ == "__main__":
    main()