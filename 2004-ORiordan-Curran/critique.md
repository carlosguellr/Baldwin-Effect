# summary

## what it does

The paper explores the impact of lifetime learning in changing environments. A population of 200 neural networks is evolved by a genetic algorithm where network is encoded as a bit string using marker based encoding. Agents are tasked to classifcy 16 four bit patterns as food or poison (4 bit parity problem)

In the learning trials, agents get backpropagation cycles before the fitness is measured, and the genome is not changed in that training- whereas the no-learning trials get none. The environment either changes gradually, with one pattern flipped every 19 generations, or suddenlt, where all patterns flip at generation 130. The test is averaged over 20 runs

## why it demonstrates the baldwin effect

Learning is plastic and non-inherited, since offspring get the original genome (network structure) but not the phenes (learned weights). Selection acts on post-learning fitness, so learning does impact which genomes will get selected. 

Furthermore, the learning populations reach overall higher fitness and have a susbtantially smaller fitness drop at the sudden flip, of about 0.15 compared to about 0.30 without learning. 

# critique

good
- clean, controllable with only difference between trials is backpropagation. Same seeds give same starting populations
- transferring structure only avoids potential lamarckiaan 



bad
- many assumptions had to be made. fitness, learning rate, cycle count, initialisation and selection not defined. Many parameters and functions taken from their later papers. 
- genetic assimilation isn't tested.
- learning has no cost - agent's just get to learn 
- parity is never solved -> looks like potential convergence 
- task is tiny, learning is more so memorisation over generalisation 


## takeaway from the experiment
establishes the mechanisms that the GA we will make reies on -> a non-inherited learning phase before fitness is measured is needed. for the GA this can be carried into hyperparameter search, where plasticity is the guessing of unfixed hyperparameter genes (?)

- fitness and initialisation can decide whether the effect appears at all

- vary amount of learning, test assimilation . The experiment compares learning / no learning. ga should be able to compare amount of plasticity, and check whether later generations need fewer trials (assimilation)

- although keras was slower here because of a large population of tiny networks, it will be beneficial to work with standard models that will be trained for more epochs. amount of training should be held fixed though, unless it is used as the learning variable. 