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

# Calibrating and asexual

t=   1  pop=100
  t=  21  pop=132
  t=  41  pop=112
  t=  61  pop=95
  t=  81  pop=63
  t= 101  pop=55
  t= 121  pop=46
  t= 141  pop=74
  t= 161  pop=120
  t= 181  pop=141
  t= 201  pop=165

what happens here is that at the start there is a lot of food so the population goes up
but when that happens the tax exceeds what the income can replenish and agents start starvin
thus population goes down, then it rises again with excess food 

We raised pile_depth (fewer, much bigger piles) rather than piles_per_step (more, normal-sized piles spread around). 
Bgger, sparser piles mean food is more spatially concentrated, so a local boom of agents around one big pile can crash hard once it's exhausted, whereas more evenly distributed piles would even out supply and might produce a smoother, less oscillatory curve — closer to what the paper's figures seem to show (they don't look this jagged, though we can't read exact numbers off a printed graph).

HERE - TRY CALIBRATING PILES PER STEP INSTEAD 