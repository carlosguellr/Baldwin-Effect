import numpy as np 
from genome import GENOME_LEN, encode_node, N_IN, OUT, FIRST_HID
from network import evaluate

POP_SIZE = 200
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.002       # Assuming rate per bit
GENERATIONS = 300
GRADUAL_CYCLES = 19
SUDDEN = 150

class Environment:
    """
    The 16 possible four bit pattern where 1 = food and 0 = poison.
    """
    def __init__(self):
        self.X = np.array([[int(b) for b in format(i, "04b")] for i in range(16)], dtype=float)
        self.t = self.X.sum(axis=1) % 2                          # 4-bit parity: 1 = food, 0 = poison
        self.next_flip = 0

    def flip_one(self):
        """Flip food/poison for one of the 16 patterns, in order"""
        self.t[self.next_flip] = 1 - self.t[self.next_flip]
        self.next_flip += 1

    def flip_all(self):
        """Flip food/poison for all"""
        self.t = 1 - self.t

# Genetic algorithm functions

def linear_rank(fitnesses): 
    """Raw fitness is transformed to rank based iftness in range [0,1]
        Tied agents share their average rank.
    """
    n = len(fitnesses)
    uniq, inv, counts = np.unique(fitnesses, return_inverse=True, return_counts=True)
    avg = (np.cumsum(counts) - counts) + (counts - 1) / 2
    return avg[inv] / (n-1)

def roulette_wheel_selection(ranked, rng):
    """Select indices proportionate to ranked fitness using roulette wheel selection."""
    return rng.choice(len(ranked), size=len(ranked), p=ranked / ranked.sum())

def crossover(parent1, parent2, rate, rng):
    """Single point crossover with probability rate. Returns two children."""    
    if rng.random() < rate:
        cut = rng.integers(1, len(parent1))
        return parent1[:cut] + parent2[cut:], parent2[:cut] + parent1[cut:]
    return parent1, parent2

def mutate(genome, rate, rng):
    """Randomly flip bits in genome with probability rate."""
    bits = np.frombuffer(genome.encode(), dtype=np.uint8) - 48 # convert '0'/'1' to 0/1
    bits = bits ^ (rng.random(len(bits)) < rate).astype(np.uint8)  # flip bits with probability rate
    return (bits + 48).tobytes().decode()  # convert back to '0'/'1'

def next_generation(population, ranked, rng):
    chosen = [population[i] for i in roulette_wheel_selection(ranked, rng)]
    new = []
    for i in range(0, len(chosen), 2):
        p1, p2 = crossover(chosen[i], chosen[i+1], CROSSOVER_RATE, rng)
        new += [mutate(p1, MUTATION_RATE, rng), mutate(p2, MUTATION_RATE, rng)]
    return new 



def structured_genome(rng):
    """Random genome built from valid node records, so the output is always wired to the inputs.
    2-5 hidden nodes (each fed by 2-4 inputs, sometimes a lower hidden node), an output fed by
    1+ hidden nodes; records are shuffled and the rest of the genome is padded with random bits."""
    n_hid = int(rng.integers(2, 6))
    labels = sorted(int(l) for l in rng.choice(np.arange(FIRST_HID, OUT), n_hid, replace=False))
    w = lambda: rng.integers(-8, 8) / 2          # a random value on the 4-bit grid [-4, 3.5]
    records = []
    for i, lab in enumerate(labels):
        srcs = [int(s) for s in rng.choice(N_IN, int(rng.integers(2, 5)), replace=False)]
        if i > 0 and rng.random() < 0.5:
            srcs.append(int(rng.choice(labels[:i])))
        records.append(encode_node(lab, w(), [(s, w()) for s in srcs]))
    srcs = [int(s) for s in rng.choice(labels, int(rng.integers(1, n_hid + 1)), replace=False)]
    records.append(encode_node(OUT, w(), [(s, w()) for s in srcs]))
    rng.shuffle(records)
    g = "".join(records)
    return g + "".join(map(str, rng.integers(0, 2, GENOME_LEN - len(g))))

def random_bits_genome(rng):
    return "".join(map(str, rng.integers(0, 2, GENOME_LEN )))

def run(mode, cycles, seed=0, generations=GENERATIONS, pop_size=POP_SIZE, learning_rate=0.3, init="random"):
    """
        One run of the algorithm. Mode is either "gradual" or "sudden".
        Cycles: backpropagations cycles per lifetime (0 = no learning)
        Returns mean raw fitness per generation, best raw fitness per generation.
    """
    rng = np.random.default_rng(seed)
    env = Environment()
    make = structured_genome if init == "structured" else random_bits_genome
    population = [make(rng) for _ in range(pop_size)]
    mean_fitnesses, best_fitnesses = [], []
    for gen in range(generations):
        if mode == "gradual" and gen > 0 and gen % GRADUAL_CYCLES == 0 and env.next_flip < 16:
            env.flip_one()
        elif mode == "sudden" and gen == SUDDEN:
            env.flip_all()
        raw_fitnesses = np.array([evaluate(g, env.X, env.t, cycles, learning_rate) for g in population])
        mean_fitnesses.append(raw_fitnesses.mean())
        best_fitnesses.append(raw_fitnesses.max())
        population = next_generation(population, linear_rank(raw_fitnesses), rng)
    return np.array(mean_fitnesses), np.array(best_fitnesses)


if __name__ == "__main__":
    import time 
# for cycles in (0, 20):
    t0 = time.time()
    m, b = run("sudden", 50, generations=300, learning_rate=3.0)
    print(f"cycles={50:2d}  {time.time() - t0:5.1f}s  mean@0={m[0]:.3f} mean@59={m[-1]:.3f}  best@59={b[-1]:.3f}")