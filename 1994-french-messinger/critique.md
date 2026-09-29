# summary



## what it does



## why it demonstrates the baldwin effect



# critique

good
- as a demonstration of baldwin effect it is "better" than hinton and nowlan - instead of specifying a fitness formula they let fitness appear organically through the agents behavior 
- 


bad
- application to machine learning process --> the findings of the experiment are tightly knit with the implementation and agentic behavior 
- found a lot of issues in the experiment such as population cycles, seed sensitive extinction --> interesting for ecology but will create noise for NNs where we are looking for convergence and stability


## agentic behavior and takeaway from the experiment

what transfers
- plasticity curve (also in hinton and Nowlan) --> the idea that some parameters are resolved by a search per individual with a per parameter difficulty or plasticity parameter that controls how likely it is to succeed 
    so architecture choices fixed genetically but some parameters tuned by a per candidate search before evaluating
- the idea of moderate plasticity being the best. can be interesting to explore in future GA whether a small amount of per individual local search is more helpful than none or too much (is too much just a random search?)

what doesn't
- the entire agentic idea --> neural nets already give an explicit fitness function which is their accuracy (MAE, MSQ)
- spatial grid with food --> even if you try to implement an implicit fitness for NNs it will end up with an explicit formula involved even if you try to say tax food based on MAE 
