# summary



## what it does
- simulate a population of agents in a 100x100 grid who eat and pay energy taxes for movement, metabolism and reproduction. they reproduce asexually when they have stored enough food (which is randomly distributed on the grid every so often)
- there is no explicit fitness curve, it is implicit in the behavior of the agents 
- agents have a 12 bit sequence representing its genotype, there is a particular sequence which is deemed the GG and acquiring it (naturally or through lifetime learning, its difficulty based on a plasticity variable and the hamming distance from the GG bit sequency) provides the GP. GP gives a tax levy on either movement, metabolism or reproduction. 
- compare population with learning vs no learning (plasticity off)


## why it demonstrates the baldwin effect
- Without learning the fitness (GG & GP population) is flat, and the only way to acquire the GP is by a chance mutation (needle in a haystack), and spread through natural evolution
- With learning, even agents far from the GG have a chance to acquire the GP. 
- This is shown --> with learning the mean pop Hamming distance to GG declines from about 0.5 to 0.2, while no learning stays flat and noise
- Genetic assimilation can also be seen (GP and GG %s converge and rise once the gene is discovered, no longer phenotype dependent)

# critique

good
- as a demonstration of baldwin effect it is "better" than hinton and nowlan - instead of specifying a fitness formula they let fitness appear organically through the agents' survival behavior rather than as a byproduct of the GA's fitness design 
- Reproduction is asynchronous causing continuous generations which is closer to a biological population rather than being syncrhonized for the purpose of the GA
- Demonstrates genetic assimilation 


bad
- application to machine learning process --> the findings of the experiment are tightly knit with the implementation and agentic behavior 
- found a lot of issues in the experiment such as population cycles, seed sensitive extinction --> interesting for ecology but will create noise for NNs where we are looking for convergence and stability
- The timing of when the Good Gene is discovered is random — single runs are close to meaningless, and only averaging over many seeds reveals a pattern. (expensive for HPO)
- there's no clean way to separate "this genotype is better" from "this agent got lucky with food"

- reproduction success depends on crowding rather than fitness --> a child can only be placed on an empty cell if all four cells adjacent to the parent are free - it can fail regardless how good the genotype is. 
- for sexual reproduction it is the same --> mates have to be adjacent, so finding a mate becomes spatial dependent rather than based on genotype. an agent with a better genotype might die out if it is isolated


## two sources of randomness
1. genotype --> driven by mutation rng
2. whether reproduction happens at all and whose genes (which agent) gets a chance --> spatial RNG
reproductive success becomes - did someone wander next to me + do i happen to have food

this is likely what causes the population cycles
to dampen randomness i averaged out over seeds



## agentic behavior and takeaway from the experiment

what transfers
- plasticity curve (also in hinton and Nowlan) --> the idea that some parameters are resolved by a search per individual with a per parameter difficulty or plasticity parameter that controls how likely it is to succeed 
    so architecture choices fixed genetically but some parameters tuned by a per candidate search before evaluating
- the idea of moderate plasticity being the best. can be interesting to explore in future GA whether a small amount of per individual local search is more helpful than none or too much (is too much just a random search?)

what doesn't
- the entire agentic idea --> neural nets already give an explicit fitness function which is their accuracy (MAE, MSQ)
- spatial grid with food --> even if you try to implement an implicit fitness for NNs it will end up with an explicit formula involved even if you try to say tax food based on MAE 
