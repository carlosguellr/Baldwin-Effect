"""
Recreation of French and Messinger's 1994 Paper "Genes, Phenes and the Baldwin Effect"
"""

from dataclasses import dataclass

import numpy as np 
import matplotlib.pyplot as plt

def plasticity_prob(d, rho):
    """
        Probability of plasticity as a function of distance d and parameter rho.
        Paper: y = 1 - d**rho. d = 0 always gives 1 (the agent has the Good Gene).
        Small rho -> almost nobody learns; large rho -> almost everybody does.
        d -> hamming distance from the good phene 
    """
    d = np.asarray(d, dtype=float)
    return 1 - np.power(d, rho)

def plot_plasticity(show=True):
    """
        Plot the plasticity probability function for different values of rho.
    """

    x = np.linspace(0, 1, 501)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))

    #fig 1: rho = 0.1 
    a1.plot(x, plasticity_prob(x, 0.1))
    a1.scatter([0.2], [plasticity_prob(0.2, 0.1)], color='red', zorder=3)
    a1.annotate(f"Plasticity probability at d=0.2: {plasticity_prob(0.2, 0.1):.2f}", 
                (0.2, plasticity_prob(0.2, 0.1)), xytext=(0.35, 0.3))
    a1.set_title("Fig 1. Plasticity probability for rho=0.1")

    #fig 2: family of curves (rho 0.01, 0.2, 1.5, 6.0)
    for rho in [0.01, 0.2, 1.5, 6.0]:
        a2.plot(x, plasticity_prob(x, rho), label=f"rho={rho}")
    a2.legend()
    a2.set_title("Fig 2. Plasticity probability for varying rho values")

    for ax in [a1, a2]:
        ax.set_xlabel("Normalized Hamming distance from GG")
        ax.set_ylabel("Plasticity probability (learn good gene)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    if show:
        plt.show()
    return fig

@dataclass
class Params:
    grid_size: int = 100    #100x100 grid with at most 1 agent per cell

    gene_length: int = 10  # Length of each agent's gene
    start_pop: int = 100
    start_food: int = 100
    start_piles: int = 100      #calibrated (paper was 30)
    piles_per_step: int = 4     #NumberFoodPilesTurn
    pile_depth:int = 1100       #FoodPileDepth - calibrated, was 20 in the paper
    mouthful:int = 100          #max food eaten per step

    # taxes
    meta_tax: int = 5           #tax for existing (even when not moving)
    move_tax: int = 15
    repo_tax: int = 150
    repo_threshold: int = 200

    #genetics
    mutation_rate: float = 0.005    # Paper says = 50?? what dopes that mean. PLACEHOLDER PER BIT

    #learning 
    learning:bool = True
    plasticity:float = 0.1         #rho
    gp_benefit:float = 2.0         #benefit of learning the good phene - tax is divided by this
    gp_type:int =1                 # 1 = move, 2 = metabolism, 3 = reproduce
    sexual:bool = False

    #implementation
    max_agents:int = 5000
    seed:int = 42

    # Calibration:
    # the papers food income  (piles_per_step * pile_depth = 80/step)
    # can't sustain a population of few hundred paying 5-20 tax pert step - they go extinct
    # raising pile depth settles into a stable population that matches the figures of the paper
    # start piles was raised to 100 because with 30 an unlucky run can starve the population before
    # they have the chance to catch up 

class World:
    def __init__(self, p: Params):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        n, L, g = p.max_agents, p.gene_length, p.grid_size

        self.good_gene = np.ones(L, dtype=np.uint8)  # Good gene is all ones

        #per agent arrays
        self.alive = np.zeros(n, dtype=np.bool_) 
        self.x = np.zeros(n, dtype=np.int32)
        self.y = np.zeros(n, dtype=np.int32)
        self.food = np.zeros(n, dtype=np.int32)
        self.genome = np.zeros((n, L), dtype=np.uint8)
        self.has_gp = np.zeros(n, dtype=np.bool_)
        self.last_dir = np.zeros(n, dtype=np.int32)

        #grids
        self.occ = np.full((g, g), -1, dtype=np.int32)  # agent, or -1 (= empty)
        self.food_grid = np.zeros((g, g), dtype=np.int32)  # food piles

        self.t = 0 # time step
        self._seed_world()

    def _seed_world(self):
        p, rng = self.p, self.rng
        self.add_food(p.start_piles)        #Initial food piles - assumes that piles landing on the same cell stack

        cells = rng.choice(p.grid_size**2, size=p.start_pop, replace=False)
        for slot, cell in enumerate(cells):
            self.spawn(slot, cell // p.grid_size, cell % p.grid_size, 
                       rng.integers(0, 2, size=p.gene_length, dtype=np.uint8), 
                       p.start_food)
        
    def spawn(self, slot, x, y, genome, food):
        self.alive[slot] = True
        self.x[slot] = x
        self.y[slot] = y
        self.genome[slot] = genome
        self.has_gp[slot] = False
        self.food[slot] = food
        self.occ[x, y] = slot
        self.has_gp[slot] = self.resolve_good_phene(slot)

    def resolve_good_phene(self, slot):
        """
            Decide whether agent at slot ends up with the GP
            d == 0 --> agent has GG, always succeeds (1-0*rho == 1)
            If p.learning is false no learning happens at all, only agents with GG get GP
        """
        d = self.normalized_hamming(slot)
        if not self.p.learning:
            return d == 0.0
        p_learn = plasticity_prob(d, self.p.plasticity)
        return bool(self.rng.random() < p_learn)

    def add_food(self, n_piles):
        g = self.p.grid_size
        xs = self.rng.integers(0, g, size=n_piles)
        ys = self.rng.integers(0, g, size=n_piles)
        for x, y in zip(xs, ys):
            self.food_grid[x, y] += self.p.pile_depth


    def n_alive(self):
        return int(self.alive.sum())
    def normalized_hamming(self, slot):
        return float(np.mean(self.genome[slot] != self.good_gene))
    def n_good_gene(self):
        """Count agents whose genome exactly matches the Good Gene (d = 0)."""
        alive_idx = np.flatnonzero(self.alive)
        return int(np.sum(np.all(self.genome[alive_idx] == self.good_gene, axis=1)))
    def n_good_phene(self):
        return int(self.has_gp[self.alive].sum())


    #4 neighbor offset 
    _DX = (1, -1, 0, 0)
    _DY = (0, 0, 1, -1)

    def _neighbors(self, x, y):
        g = self.p.grid_size
        return [((x+dx)%g, (y+dy)%g) for dx, dy in zip(self._DX, self._DY)]

    def _find_free_slot(self):
        free = np.flatnonzero(~self.alive)  #~ flips the aray (True os where a slot is free) - flatnonzero returns indices where that condition is true
        return int(free[0]) if free.size else None

    def _mutate(self, genome):
        #flips decides which genes mutate (random generates a random num for each gene)
        flips = self.rng.random(genome.shape[0]) < self.p.mutation_rate
        child = genome.copy()
        child[flips] ^= 1 #XOR to flip 
        return child

    def _kill(self, slot):
        self.occ[self.x[slot], self.y[slot]] = -1
        self.alive[slot] = False

    def _reproduce_asexual(self, slot):
        p = self.p
        tax = p.repo_tax
        if self.has_gp[slot] and p.gp_type == 3: 
            tax /= p.gp_benefit     # reduce amount taxed for reproduction if GP is of type 3
        if self.food[slot] - tax < 0:
            return                  # can't afford to reproduce

        #Place child on a free cell in parent's adjacency, if not free it fails - ASSUMPTION
        x, y = self.x[slot], self.y[slot]
        empty= [(cx, cy) for cx, cy in self._neighbors(x, y) if self.occ[cx, cy] == -1]

        if not empty: 
            return 
        free_slot = self._find_free_slot()
        if free_slot is None:
            return  # max agents reached

        cx, cy = empty[self.rng.integers(0, len(empty))]
        child_genome = self._mutate(self.genome[slot])
        self.food[slot] -= tax
        self.spawn(free_slot, cx, cy, child_genome, tax)    #child inherits the taxed food

    def _move(self, slot):
        """Move toward the neighbour cell with the most food.
 
        ASSUMPTION: if the best cell is occupied, the agent stays put and
        pays no movement tax.
        """
        p = self.p
        x, y = self.x[slot], self.y[slot]
        neigh = self._neighbors(x, y)
        levels = np.array([self.food_grid[cx, cy] for cx, cy in neigh])
        best = levels.max()
        if best == levels.min():
            direction = self.last_dir[slot]  # all equal: keep going the same way
        else:
            candidates = np.flatnonzero(levels == best)
            direction = candidates[0] if self.last_dir[slot] not in candidates \
                else self.last_dir[slot]
        nx, ny = neigh[direction]
        if self.occ[nx, ny] != -1:
            return  # blocked: stay, no tax
        tax = p.move_tax
        if self.has_gp[slot] and p.gp_type == 1:
            tax /= p.gp_benefit
        self.occ[x, y] = -1
        self.x[slot], self.y[slot] = nx, ny
        self.occ[nx, ny] = slot
        self.last_dir[slot] = direction
        self.food[slot] -= tax
        if self.food[slot] <= 0:
            self._kill(slot)

    def step(self):
        p = self.p
        self.add_food(p.piles_per_step)
 
        order = self.rng.permutation(np.flatnonzero(self.alive))
        for slot in order:
            if not self.alive[slot]:
                continue
 
            # Metabolize
            tax = p.meta_tax
            if self.has_gp[slot] and p.gp_type == 2:
                tax /= p.gp_benefit
            self.food[slot] -= tax
            if self.food[slot] <= 0:
                self._kill(slot)
                continue
 
            # Eat
            x, y = self.x[slot], self.y[slot]
            eaten = min(self.food_grid[x, y], p.mouthful)
            self.food[slot] += eaten
            self.food_grid[x, y] -= eaten
 
            # Reproduce (asexual for now)
            if self.food[slot] > p.repo_threshold:
                self._reproduce_asexual(slot)
 
            # Move
            if self.alive[slot]:
                self._move(slot)
 
        self.t += 1

def run_simulation(p: Params, n_steps: int, log_every: int = 1):
    """Run the world for n_steps, logging population  GP and GG counts"""
    w = World(p)
    ts, pop, gp, gg = [], [], [], []
    for step in range(n_steps):
        w.step()
        if step % log_every == 0:
            ts.append(w.t)
            pop.append(w.n_alive())
            gp.append(w.n_good_phene())
            gg.append(w.n_good_gene())
        if w.n_alive() == 0:
            break
    return {"t": np.array(ts), "pop": np.array(pop),
            "gp": np.array(gp), "gg": np.array(gg)}

if __name__ == "__main__":
    plot_plasticity()

    p = Params()
    w = World(p)
    print(f"Initial number of alive agents: {w.n_alive()} (expected {p.start_pop})")
    print(f"Occupied cells: {int((w.occ >=0).sum())}")
    print(f"Total food on grid: {w.food_grid.sum()} (expected {p.start_piles*p.pile_depth})")
    
    d = [w.normalized_hamming(i) for i in range(p.start_pop)]
    print(f"Mean normalized Hamming d: {round(np.mean(d), 3)}   (expected 0.5 for random genomes)")


    #does the learning rate match plasticity_prob? --> build many agents at fixed distance and see
    # what fraction ends up with GP
    p3 = Params(gene_length=10, plasticity=0.1, learning=True, seed=1,
                max_agents=20000)
    w3 = World(p3)
    trials = 20000
    genome_d02 = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)  # d=0.2
    got_gp = 0
    for slot in range(trials):
        w3.spawn(slot, 0, 0, genome_d02, p3.start_food)  # same cell is fine here
        got_gp += w3.has_gp[slot]
    expected = plasticity_prob(0.2, 0.1)
    print(f"learning rate at d=0.2, rho=0.1: {got_gp/trials:.3f} "
          f"(expected ~{expected:.3f})")


    # Step 4 check: run the (uncalibrated) control for a bit and see
    # what the population does. Paper values give ~80 food/step income,
    # which is not going to sustain ~100+ agents paying tax every step --
    # expect a crash here. This sets up the calibration in Step 5.
    p4 = Params(learning=False, seed=2)
    log = run_simulation(p4, n_steps=201,log_every=20)
    print("\nuncalibrated control run (pop every 20 steps):")
    for t, pop in zip(log["t"], log["pop"]):
        print(f"  t={t:4d}  pop={pop}")