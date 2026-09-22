# Developing
The np.uint8, np.int32, np.bool_ on the arrays in World are a separate, deliberate choice: they set the storage type for the whole array (e.g. genome bits stored as 1 byte each instead of Python's ~28-byte int objects), and that's what makes the arrays cheap to store and fast to operate on — and it's a precondition for Numba later.

repo threshold  - an agent can only reproduce once its internal food exceeds this value

pile_depth and piles_per_step. Every timestep, piles_per_step new food piles are dropped at random cells, each with pile_depth food in it. So total food income per timestep = piles_per_step × pile_depth

self.occ is the occupancy grid (100x100 array) matching grid size that tracks which agent is on which cell.
self.occ[x,y] = slot means agent in the array slot slot is sitting on cell x,y
self.occ[x,y] = -1 means the cell is empty 

# Figures 1 - 2 - Impact of Plasticity and Hamming Distance

Those two plots aren't simulation output at all — they're just the formula 1 - d**rho evaluated across x = linspace(0, 1, 50)

# Figure 3 - Birth and Learning
only use w3 for this testing

the objective of this section is to spawn 20 000 agents all 2 bits away from GG, all having the same distance and rho of 0.1. each one independently attempts to learn the GP and we see whether we get the expected result. we had a result of 0.150 with 0.149 expected, showing that the learning logic does what the formula says it should

