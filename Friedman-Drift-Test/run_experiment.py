"""
Runs NEAT on the drifting Friedman environment and saves one CSV per (condition, seed).

darwinian : weights are fixed at birth, no lifetime learning.
            Each genome is scored as-is on the validation set.

baldwinian : same as darwinian, but each genome first learns on the training set
             (LEARN_STEPS Adam steps) and is then scored on the validation set
             with the learned weights. The learned weights are thrown away:
             offspring inherit the birth weights only.

lamarckian : same as baldwinian, but the learned weights are inherited by the offspring
             (not implemented yet, will use GenomeNet.write_to_genome).


Per generation, each run logs the BEST genome of that generation:
    val_MAE              validation MAE after learning (what selection used, i.e. -fitness)
    test_MAE             MAE on the held-out test set of the current concept, after learning
    innate_test_MAE      test MAE with the birth weights, before any lifetime learning
                         (identical to test_MAE for darwinian; for baldwinian, this
                         falling over generations is the Baldwin effect)
    pop_mean_val_MAE     mean validation MAE of the whole population
    hidden_nodes         number of hidden nodes in the best genome
    enabled_connections  number of active connections in the best genome
    n_params             number of trainable parameters in the best genome
    n_species            number of species in the population
    in_x0 .. in_x9       1 if input xi has an enabled path to the output in the best genome
    pop_in_x0 .. x9      fraction of the population with an enabled path from xi to the output


outputs results in results/<condition>/seed_<NN>.csv
"""
import os
import random
import time
from multiprocessing import Pool

import neat
import numpy as np
import pandas as pd

from friedman_data import CONCEPTS, DRIFT_GENS, GENERATIONS, block_for_generation, make_blocks
from learning import GenomeNet

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "neat_config.ini")
RESULTS_DIR = os.path.join(HERE, "results")

CONDITIONS = ["darwinian", "baldwinian"]
N_SEEDS = 20              # runs seeds 0 .. N_SEEDS-1 for every condition
WORKERS = os.cpu_count()   # parallel processes; set to 1 to run one after another
LEARN_STEPS = 20           # lifetime learning: Adam steps per genome
LEARN_RATE = 0.05          # lifetime learning: Adam step size

OUTPUT_ACTIVATION = "identity" 
class FixedOutputGenome(neat.DefaultGenome):
    """DefaultGenome whose output node always uses OUTPUT_ACTIVATION.
    Hidden nodes can still switch between the activation_options."""

    def _fix_output(self, config):
        for key in config.output_keys:
            self.nodes[key].activation = OUTPUT_ACTIVATION

    def configure_new(self, config):
        super().configure_new(config)      # initial population
        self._fix_output(config)

    def mutate(self, config):
        super().mutate(config)             # every offspring is mutated after crossover
        self._fix_output(config)

# ---------- evaluation ----------
def evaluate_genome(genome, config, block, condition):
    """Score one genome on the current concept block.

    Returns (net, theta, val_mae), where theta are the weights actually scored:
    the birth weights (darwinian) or the learned weights (baldwinian).
    """
    net = GenomeNet(genome, config)
    if condition == "darwinian":
        theta = net.theta0
    elif condition == "baldwinian":
        # Learning uses the training set only; the genome itself is not modified
        theta = net.learn(block["train"], steps=LEARN_STEPS, lr=LEARN_RATE)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return net, theta, net.mae(theta, block["val"])


def make_eval_function(state, condition):
    """Build the fitness function neat-python calls once per generation.

    ``state["block"]`` holds the concept block. It is updated by the
    run loop when the environment drifts, so the same function always scores
    genomes on the current concept.

    The best genome of this generation is stored in ``state["gen_best"]``, because
    ``Population.run`` returns the best genome EVER seen, which after a drift
    can be a stale genome scored on the old concept.
    """
    def eval_genomes(genomes, config):
        scored = []
        for _, genome in genomes:
            net, theta, val_mae = evaluate_genome(genome, config, state["block"], condition)
            # Selection uses the validation set, never the test set
            genome.fitness = -val_mae
            scored.append((val_mae, genome, net, theta))

        _, *state["gen_best"] = min(scored, key=lambda s: s[0])
        state["pop_mean_val_MAE"] = float(np.mean([s[0] for s in scored]))
        state["pop_input_frac"] = np.mean([s[2].input_mask for s in scored], axis=0)
    return eval_genomes


def run_seed(condition, seed, verbose=False):
    """Run one full evolutionary run and save its per-generation log.

    The seed fixes both the data and NEAT's random choices, so the same seed
    gives the same data in every condition.
    """
    random.seed(seed)          # neat-python uses Python's random module
    np.random.seed(seed)
    blocks = make_blocks(seed)

    config = neat.Config(FixedOutputGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)
    pop = neat.Population(config)

    state = {"block": None, "gen_best": None, "pop_mean_val_MAE": None, "pop_input_frac": None}
    eval_genomes = make_eval_function(state, condition)

    log, t0 = [], time.time()
    for g in range(GENERATIONS):
        b = block_for_generation(g)
        state["block"] = blocks[b]                 # switch concept at the drift points

        # At a drift, NEATs best fitnesses refers to the previous concept
        # so every species would look stagnant. 
        # Clear the history so improvement is judged on new concept.
        if g in DRIFT_GENS:
            for species in pop.species.species.values():
                species.fitness_history = []

        pop.run(eval_genomes, 1)                   # evolve exactly one generation

        genome, net, theta = state["gen_best"]
        test = blocks[b]["test"]
        log.append({
            "seed": seed,
            "condition": condition,
            "generation": g,
            "concept": CONCEPTS[b],
            "val_MAE": -genome.fitness,
            "test_MAE": net.mae(theta, test),
            "innate_test_MAE": net.mae(net.theta0, test),
            "pop_mean_val_MAE": state["pop_mean_val_MAE"],
            "hidden_nodes": len(genome.nodes) - config.genome_config.num_outputs,
            "enabled_connections": sum(c.enabled for c in genome.connections.values()),
            "n_params": len(net.theta0),
            "n_species": len(pop.species.species),
            # Input connectivity: best genome (0/1) and population fraction, per input
            **{f"in_x{i}": int(m) for i, m in enumerate(net.input_mask)},
            **{f"pop_in_x{i}": float(f) for i, f in enumerate(state["pop_input_frac"])},
        })
        if verbose and (g % 25 == 0 or g == GENERATIONS - 1):
            r = log[-1]
            print(f"[{condition} {seed}] gen {g:3d} [{r['concept']}]  val {r['val_MAE']:.2f}  "
                  f"test {r['test_MAE']:.2f}  innate {r['innate_test_MAE']:.2f}  "
                  f"hidden {r['hidden_nodes']}  ({time.time() - t0:.0f}s)")

    out_dir = os.path.join(RESULTS_DIR, condition)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed_{seed:02d}.csv")
    pd.DataFrame(log).to_csv(path, index=False)
    print(f"[{condition} {seed}] done in {time.time() - t0:.0f}s -> {path}")
    return path


def main():
    """Run every (condition, seed) pair, in parallel unless WORKERS is 1."""
    jobs = [(c, s) for c in CONDITIONS for s in range(N_SEEDS)]
    t0 = time.time()
    if WORKERS == 1:
        for c, s in jobs:
            run_seed(c, s, verbose=True)
    else:
        # Each run is independent, so runs execute in parallel processes.
        with Pool(WORKERS) as pool:
            pool.starmap(run_seed, jobs)
    print(f"All {len(jobs)} runs finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
