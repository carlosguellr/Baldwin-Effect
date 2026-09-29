import numpy as np 
import matplotlib.pyplot as plt

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

def fitness_no_learning(population, rng=None):
    all_correct = (population == 1).all(axis=1)
    return np.where(all_correct, 20.0, 1.0)

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

def run_simulation(seed, learning = True):
    """Run a full 51 generation simulation and return its history array"""
    local_rng = np.random.default_rng(seed)
    population = local_rng.choice([0,1,2], size=(POP_SIZE,N_GENES), p=[0.25, 0.25, 0.5])   #Alleles with 2 = "?"
    history = []
    for generation in range(GENS):
        fitness_values = fitness(population, local_rng) if learning else fitness_no_learning(population)
        history.append([(population == 1).mean(),
                        (population == 0).mean(),
                        (population == 2).mean(),
                        fitness_values.mean()])
        population = next_generation(population, fitness_values, local_rng)
    return np.array(history)

def run_experiment(n_runs, base_seed=0, learning = True):
    return np.stack([run_simulation(base_seed+i, learning = learning) for i in range(n_runs)])

N_RUNS = 20
results = run_experiment(N_RUNS, learning=True)    # (N_RUNS, GENS, 4)
mean_hist = results.mean(axis=0)          # (GENS, 4)
std_hist  = results.std(axis=0)           # (GENS, 4)
results_no_learning = run_experiment(N_RUNS, learning=False)

generations = np.arange(GENS)


#Plot 2: mean fitness learning vs no learning
mean_learn = results.mean(axis=0)[:, 3]
mean_no_learn = results_no_learning.mean(axis=0)[:, 3]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(generations, mean_learn, label="With learning (plastic alleles)")
ax.plot(generations, mean_no_learn, label="No learning (fixed alleles)")
ax.fill_between(generations,
                np.clip(mean_hist[:, 3] - std_hist[:, 3], 1.0, None),
                np.clip(mean_hist[:, 3] + std_hist[:, 3],1.0, None),
                 color="C0", alpha=0.2, label="±1 std across runs")
ax.set_xlabel("Generation")
ax.set_ylabel("Mean population fitness")
ax.set_title(f"Population fitness (across {N_RUNS} runs)")
ax.legend()
fig.tight_layout()


#Plot 3: Evolution of population allele composition
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(generations, mean_hist[:, 0], label="Correct (1)", color="tab:green")
ax.plot(generations, mean_hist[:, 1], label="Incorrect (0)", color="tab:red")
ax.plot(generations, mean_hist[:, 2], label="Plastic (?)", color="tab:orange")
ax.set_xlabel("Generation")
ax.set_ylabel("Proportion of alleles")
ax.set_title(f"Allele composition (mean across {N_RUNS} runs)")
ax.legend()
fig.tight_layout()


plt.show()