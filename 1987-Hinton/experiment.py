import numpy as np 
POP_SIZE = 1000
N_GENES = 20
TRIALS = 1000
GENS = 51

def fitness(population, rng):
    incorrect_alleles = (population == 0).any(axis=1)   # Individuals with at least one incorrect allele, defining any that isn't all 1s as wrong
    q = (population == 2).sum(axis=1)                   # Count of total "?" alleles
    first_success_trial = rng.geometric(2.0**-q)        # (blind guessing) - assumes each ? gene with 50% chance of being correct, so 2^-q is the probability of success for an individual with q "?" alleles
    trials_remaining = np.where(first_success_trial <= TRIALS, TRIALS-first_success_trial, 0) # if first success occurs within TRIALs, use the remaining num, otherwise 0
    fitness = 1 + 19*trials_remaining/TRIALS            # Paper's fitness formula
    return np.where(incorrect_alleles, 1.0, fitness)    # Fitness is 1 for incorrect individuals, otherwise calculated fitness

def next_generation(population, fitness_values, rng):
    """
        Produces the next generation via fitness proportionate selection and 
        single-point selection.

        Each individual's fitness is normalized. An individual's chance of being picked
        as a parent is proportional to its fitness relative to the rest of the population
        (Best is sampled more often, but everone retains nonzero odds)

        Each offspring gets one random cut point in [1, N_GENES-1]. 
        Genes before the cut come from parent_1, genes from the cut onward come
        from parent_2 — a single contiguous swap per offspring.
    """
    probabilities = fitness_values/fitness_values.sum()  # Normalize fitness to get selection probabilities
    parent_1 = population[rng.choice(POP_SIZE, POP_SIZE, p=probabilities)]
    parent_2 = population[rng.choice(POP_SIZE, POP_SIZE, p=probabilities)]
    crossover = rng.integers(1, N_GENES, size=POP_SIZE)[:, None]
    return np.where(np.arange(N_GENES) < crossover, parent_1, parent_2)  # Crossover to create new generation


# history = np.array(history)
# print(f"{'Generation':>10} {'Correct':>8} {'Incorrect':>10} {'Plastic':>8} {'Fitness':>8}")
# for generation in range(0, GENS, 5):
#     correct, incorrect, plastic, mean_fitness = history[generation]
#     print(f"{generation:>10} {correct:>8.3f} {incorrect:>10.3f} {plastic:>8.3f} {mean_fitness:>8.2f}")


def run_simulation(seed):
    """Run a full 51 generation simulation and return its history array"""
    local_rng = np.random.default_rng(seed)
    population = local_rng.choice([0,1,2], size=(POP_SIZE,N_GENES), p=[0.25, 0.25, 0.5])   #Alleles with 2 = "?"
    history = []
    for generation in range(GENS):
        fitness_values = fitness(population, local_rng)
        history.append([(population == 1).mean(),
                        (population == 0).mean(),
                        (population == 2).mean(),
                        fitness_values.mean()])
        population = next_generation(population, fitness_values, local_rng)
    return np.array(history)

def run_experiment(n_runs, base_seed=0):
    return np.stack([run_simulation(base_seed+i) for i in range(n_runs)])

N_RUNS = 20
results = run_experiment(N_RUNS)    # (N_RUNS, GENS, 4)
mean_hist = results.mean(axis=0)          # (GENS, 4)
std_hist  = results.std(axis=0)           # (GENS, 4)

final_fitness = results[:, -1, 3]
print(f"Final-gen fitness across {N_RUNS} runs: {final_fitness.mean():.2f} ± {final_fitness.std():.2f}")
