## Why I didn't use Scikit Learn


1. Topology

MLPClassifier takes hidden_layer_sizes and will connect every node in a layer to every node in the next one. 

The random topology that is being used for the experiment cannot be represented in sklearn. 


2. Genome Weights

fit reinitialises weights unless using warm_start. Even so, backprop would update weights that would be absent (0s in the matrix, see 1. about topoligy) --> sparse structure is lost. 


3. Control over cycles

0 cycles VS N cycles (amount of learning) will mean added complexity in sklearn to change "amount" of backprop


4. speed

200 agents * 300 generations * 20 runs * 2 learn/nolearn = 2.4 million networks

sklearn overhead would not be worthwhile.