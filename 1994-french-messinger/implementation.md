## food calibration issue 
when implementing the paper some variables at the end were not given enough information to recreate the experiment, and the population crashed within a few generatins when using them. The food and amount was calibrated to allow more agents to exist. 
  ### Calibration:
     the papers food income  (piles_per_step * pile_depth = 80/step)
     can't sustain a population of few hundred paying 5-20 tax pert step - they go extinct
     raising pile depth settles into a stable population that matches the figures of the paper
     start piles was raised to 100 because with 30 an unlucky run can starve the population before
     they have the chance to catch up 

rather than raising start piles and pile depth alternative would have been to not increase pile depth as much and instead spread piles_per_step more evenly instead of randomly distributing food in big lumps

## GG% was higher without learning

Finding the exact GG by chance should be impossible within the simulated time, which should force the population to stay flat without learning. However with a gene length of 10 and a 0.5% mutation per bit it is possible to get lucky across the 150-200 generations. 

Tax reduction from gp=1 (move) applies to any agent with the GG (learning on or off)
So a lucky match gets a survival edge and spreads through the population on its own.

This will look identical to the Baldwin effect but it is instead ordinary Darwinian selection (a lucky fluke and the lineage outcompetes everyone). With learning the plasticity curve means that there should be a smooth continuous selection across the entire genotype space. Even at d=0.5 the agent has a small chance of getting the good phene via learning. so instead of a spike it is a hill that selection can climb 


Can be fixed by reducing the possibility of a lucky hit (increasing gene length to 12 and averaging out across multiple seeds)


# sexual reproduction
Assumptions made for implementation:
- Mate selection --> random mate amongst neighbors
- Mate might later be chosen by a different neighbor
- Child placement rules are identical to asexual