# summary

Hinton and Nowlan's 1987 paper *How Learning Can Guide Evolution* sets out to demonstrate the Baldwin effect, showing that learning influences the direction of evolution, without learned traits being directly inherited.

## what it does

The paper models individuals with a sequence of 20 genes, represented as either 0, 1, or ?, where 0 and 1 are fixed "alleles" and ? represents a learnable trait.

Fitness was determined by the number of trials it required to find the correct combination. Individuals that found the solution more quickly received higher fitness, while those requiring more trials had a lower fitness. Probability of reproduction was weighted by the fitness, such that more fit genotypes had a higher probability of breeding, and thus were favoured by natural selection. Over generations the ? values were increasingly replaced by genetically fixed 0s and 1s, demonstrating genetic assimilation.

## why it demonstrates the baldwin effect

- learning allows individuals to adapt to the environment + 
- baldwin effect logic - learning doesn't actually change the genotype but rather affects which genotypes tend to surive reproduction reshaping overall genotype search landscape (no risk of Lamarckian transfer of traits into genes)
- the successful adaption creates individuals with favourable genotypes that make the learned behavior (phenotype) more easily acquired  --> they are more fit generation over generation (fig 1)
- over generations, traits that initially required learning become increasingly genetically encoded (see figure 2),  can be seen in the first 20 generations. Furthermore can see a drop in incorrect alleles to 0. However incorrect alleles (?) don't fully disappear but rather plateu because a few learning trials gives them ample time to learn to adapt into the GP

# critique
good
- minimal mechanism - explores only one variable (learning/not learning), allowing the effect to be attributed to learning
- explicitly excludes lamarckian inheritance - explores pure Darwinian process that looks Lamarckian from a distance (apparently part of Baldwin's original argument)
- results were easily reproducible and not dependent on Hinton and Nowlan's exact implementation (easily testable and repeatable)


bad
- it has no "real" learning - learning is random guessing without taking an advantage of the search space. the authors themselves note a more sophisticated procedure (eg hillclimbing) could be used instead 
- only guesses - only genetic algorithm - Evolution is an abstract GA search. Connections, mutation, and selection are all disembodied from any actual organism or ecology. Worth contrasting explicitly with French & Messinger's ecological mode. where the same core idea is embedded in agents that actually move, eat, and reproduce in space
- further previous: space is nopt taken into account
- The parent-selection rule (`1 + 19n/1000`, giving instant learners a ~20× advantage) is a fairly specific, somewhat arbitrary design choice — it's not obvious this exact weighting is principled rather than tuned to make the effect visible within just 50 generations.

