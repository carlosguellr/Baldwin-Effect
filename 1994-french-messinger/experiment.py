from dataclasses import dataclass

import numpy as np 
import matplotlib.pyplot as plt

def plasticity_prob(d, rho):
    """
        Probability of plasticity as a function of distance d and parameter rho.
        Paper: y = 1 - d**rho. d = 0 always gives 1 (the agent has the Good Gene).
        Small rho -> almost nobody learns; large rho -> almost everybody does.
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
    start_piles: int = 30
    piles_per_step: int = 4     #NumberFoodPilesTurn
    pile_depth:int = 20         #FoodPileDepth
    mouthful:int = 100          #max food eaten per step

    # taxes
    meta_tax: int = 5
    move_tax: int = 15
    repo_tax: int = 150
    repo_threshold: int = 300

    #genetics
    mutation_rate: float = 0.005    # Paper says = 50?? what dopes that mean. PLACEHOLDER PER BIT

    #learning 
    learning:bool = True
    plasticity:float = 0.1         #rho
    gp_benefit:float = 2.0         #benefit of learning the good gene - tax is divided by this
    gp_type:int =1                 # 1 = move, 2 = metabolism, 3 = reproduce
    sexual:bool = False

    #implementation
    max_agents:int = 5000
    seed:int = 42

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


if __name__ == "__main__":
    plot_plasticity()

    p = Params()
    w = World(p)
    print(f"Initial number of alive agents: {w.n_alive()} (expected {p.start_pop})")
    print(f"Occupied cells: {int((w.occ >=0).sum())}")
    print(f"Total food on grid: {w.food_grid.sum()} (expected {p.start_piles*p.pile_depth})")
    
    d = [w.normalized_hamming(i) for i in range(p.start_pop)]
    print(f"Mean normalized Hamming d: round(np.mean(d), 3)   (expectAed -0.5 for random genomes)")