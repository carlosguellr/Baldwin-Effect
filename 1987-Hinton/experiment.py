import numpy as np 
rng = np.random.default_rng(42)

POP_SIZE = 1000
N_GENES = 20
TRIALS = 1000
GENS = 51

population = rng.choice([0,1,2], size=(POP_SIZE,N_GENES), p=[0.25, 0.25, 0.5])   #Alleles where 2 = "?"

def fitness(population):
    incorrect_alleles = (population == 0).any(axis=1)   # Individuals with at least one incorrect allele, defining any that isn't all 1s as wrong
    q = (population == 2).sum(axis=1)                   # Count of "?" alleles
    first_success_trial = rng.geometric(2.0**-q)        # (blind guessing) - assumes each ? gene with 50% chance of being correct, so 2^-q is the probability of success for an individual with q "?" alleles
    trials_remaining = np.where(first_success_trial <= TRIALS, TRIALS-first_success_trial, 0)
    fitness = 1 + 19*trials_remaining/TRIALS

    return np.where(incorrect_alleles, 1.0, fitness)    # Fitness is 1 for incorrect individuals, otherwise calculated fitness

def next_generation(population, fitness_values):
    probabilities = fitness_values/fitness_values.sum()  # Normalize fitness to get selection probabilities
    parent_1 = population[rng.choice(POP_SIZE, POP_SIZE, p=probabilities)]
    parent_2 = population[rng.choice(POP_SIZE, POP_SIZE, p=probabilities)]
    crossover = rng.integers(1, N_GENES, size=POP_SIZE)[:, None]
    return np.where(np.arange(N_GENES) < crossover, parent_1, parent_2)  # Crossover to create new generation


history = []
for generation in range(GENS):
    fitness_values = fitness(population)
    history.append([(population == 1).mean(),
                    (population == 0).mean(),
                    (population == 2).mean(),
                    fitness_values.mean()])
    population = next_generation(population, fitness_values)

history = np.array(history)
print(f"{'Generation':>10} {'Correct':>8} {'Incorrect':>10} {'Plastic':>8} {'Fitness':>8}")
for generation in range(0, GENS, 5):
    correct, incorrect, plastic, mean_fitness = history[generation]
    print(f"{generation:>10} {correct:>8.3f} {incorrect:>10.3f} {plastic:>8.3f} {mean_fitness:>8.2f}")