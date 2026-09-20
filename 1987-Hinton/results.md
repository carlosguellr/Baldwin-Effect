Generation  Correct  Incorrect  Plastic  Fitness
         0    0.245      0.252    0.503     1.01
         5    0.272      0.224    0.504     1.29
        10    0.575      0.000    0.425    13.03
        15    0.624      0.000    0.376    15.82
        20    0.654      0.000    0.346    17.45
        25    0.679      0.000    0.321    17.94
        30    0.696      0.000    0.304    18.41
        35    0.705      0.000    0.295    18.63
        40    0.715      0.000    0.285    18.84
        45    0.726      0.000    0.274    19.03
        50    0.733      0.000    0.267    19.05


points
    incorrect alleles go down (froom 25 to 0% in 10 generations)
        this means that theres no fixed wrong alleles that prevent learning the solution
    
    correct alleles rise from 25 to almost 75 per cent 

    plastic alleles fall from 50 to 27 per cent but never reach 0
        baldwin effect - learning guides evolution but once an individual has a few ?s then guessing is cheap
        and doesnt fix the last ones 
        (paper reports same plateu rather than full genetic assimilation)

    fitness rises from 1 to 19, close to the max of 20

jump from gens 5 to 10
    fitness is flat at first because nobody can find the target. 
    
    Only about 0.3% of random individuals (0.75^20) have no fixed 0s, and those typically have too many ? to succeed within 1000 trials. Once a few lucky individuals learn it, their offspring spread fast, and the effect takes off. That matches the paper's 