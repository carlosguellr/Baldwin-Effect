"""
This file consists of the network class, which will be used to build a network from a genome. 
The network class will also be used to evaluate the performance of the network on the parity test. 
"""

import numpy as np
from genome import decode, N_IN, OUT

N_LABELS = OUT + 1  # number of nodes in the network (input + hidden + output)
DECISION_MARGIN = 1e-6    # raw fitness output must be at least this far on the correct side of 0.5 

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class Network:
    def __init__(self, nodes):
        self.W = np.zeros((N_LABELS, N_LABELS))  # weight matrix
        self.M = np.zeros((N_LABELS, N_LABELS))  # 1 where connection exists
        self.thresholds = np.zeros(N_LABELS)  # threshold for each node
        self.order = sorted(nodes)
        self.has_output = OUT in nodes
        for label, (thr, conns) in nodes.items():
            self.thresholds[label] = thr
            for src, w in conns:
                if src < N_IN or src in nodes:  # ignore links to undefined nodes
                    self.W[src, label] = w
                    self.M[src, label] = 1

    def forward(self, X):
        """"
        Return activations A of all nodes in the network using:
            sigmoid(sum(weight*input) - threshold)
        """
        A = np.zeros((len(X), N_LABELS))
        A[:, :N_IN] = X
        for label in self.order:
            A[:, label] = sigmoid(A @ self.W[:, label] - self.thresholds[label])
        return A

    def output(self, X):
        """Return output node activations for input X."""
        if not self.has_output:
            return np.full(len(X), 0.5)  # constant output if no output node
        return self.forward(X)[:, OUT]

    def train(self, X, t, cycles, learning_rate = 0.3):
        """Train the network using backpropagation for cycles iterations. Only updates existing connections in the network."""
        if not self.has_output:
            return  # cannot train if there's no output node

        n = len(X)
        for _ in range(cycles):
            A = self.forward(X)
            delta = np.zeros_like(A)
            out = A[:, OUT]
            delta[:, OUT] = (out - t) * out * (1 - out)  # derivative of sigmoid
            for label in reversed(self.order):
                if label == OUT:
                    continue
                # collects delta of every node that has a connection to the current, multiplying by conn weight.
                delta[:, label] = (delta @ self.W[label, :]) * A[:, label] * (1 - A[:, label])

            # multiplies for every pair of nodes, source and target delta, summed over all samples, 
                #  and then multiplies by the learning rate.
            # self.M ensures that only weights with existing connections are updated (0 otherwise.)    
            self.W -= learning_rate * (A.T @ delta / n) * self.M
            self.thresholds += learning_rate * delta.mean(axis=0)  # update thresholds

def raw_fitness(net, X, t):
    return np.mean(np.abs(net.output(X) - t) < 0.5 - DECISION_MARGIN)

def evaluate(genome, X, t, cycles=0, learning_rate=0.3):
    nodes = decode(genome)
    net = Network(nodes)
    if cycles > 0:
        net.train(X, t, cycles, learning_rate)
    return raw_fitness(net, X, t)

if __name__ == "__main__":
    from genome import random_genome
    X = np.array([[int(b) for b in format(i, "04b")] for i in range(16)], dtype=float)
    t = X.sum(axis=1) % 2                          # 4-bit parity: 1 = food, 0 = poison
    genomes = [random_genome() for _ in range(300)]
    for c in (0, 20, 200):
        f = [evaluate(g, X, t, cycles=c) for g in genomes]
        print(f"cycles={c:3d}  mean fitness={np.mean(f):.3f}  max={np.max(f):.3f}")