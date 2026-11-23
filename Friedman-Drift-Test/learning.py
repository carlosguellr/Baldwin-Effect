"""
Lifetime learning for NEAT genomes: a differentiable copy of a NEAT network
plus a short burst of gradient descent on it.

neat.nn.FeedForwardNetwork evaluates one sample at a time in plain Python and
cannot compute gradients. GenomeNet builds the SAME network (same nodes,
connections, activations, biases and responses, evaluated in the same order) but
    - evaluates all samples at once with NumPy (much faster)
    - is written with autograd.numpy, so autograd can differentiate the loss
      with respect to every weight and bias

All parameters (enabled connection weights + node biases) live in one flat
vector theta. GenomeNet remembers which entry belongs to which connection / node,
so learned values can be written back into a genome (for Lamarckian inheritance).

learning : Adam, full batch, on the training set, minimising MSE on the scaled target.
           Parameters are clipped to the genome's allowed range after every step,
           as NEAT mutation would.
"""
import autograd.numpy as anp
import numpy as np
from autograd import grad
from neat.nn import FeedForwardNetwork

from friedman_data import Y_CENTRE, Y_SCALE


# ---------- activation functions (must match neat/activations.py exactly) ----------
ACTIVATIONS = {
    "sigmoid":  lambda z: 1.0 / (1.0 + anp.exp(-anp.clip(5.0 * z, -60.0, 60.0))),
    "tanh":     lambda z: anp.tanh(anp.clip(2.5 * z, -60.0, 60.0)),
    "sin":      lambda z: anp.sin(anp.clip(5.0 * z, -60.0, 60.0)),
    "gauss":    lambda z: anp.exp(-5.0 * anp.clip(z, -3.4, 3.4) ** 2),
    "relu":     lambda z: anp.maximum(z, 0.0),
    "identity": lambda z: z,
}

# Adam hyper-parameters (standard defaults)
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


class GenomeNet:
    """Vectorised, differentiable version of a NEAT feed-forward network."""

    def __init__(self, genome, config):
        gc = config.genome_config
        self.input_keys = gc.input_keys
        self.output_key = gc.output_keys[0]           # this task has a single output

        # Let neat-python decide which nodes are used and in what order,
        # so this network is guaranteed to have the same structure
        reference = FeedForwardNetwork.create(genome, config)

        theta, lo, hi = [], [], []
        self.param_keys = []    # per theta entry: ("bias", node) or ("weight", (in_node, node))
        self.plan = []          # per node: (node, activation, bias index, response, [(in_node, weight index)])

        def add_param(value, key, bounds):
            """Append one parameter to theta and return its index."""
            theta.append(value)
            self.param_keys.append(key)
            lo.append(bounds[0])
            hi.append(bounds[1])
            return len(theta) - 1

        bias_bounds = (gc.bias_min_value, gc.bias_max_value)
        weight_bounds = (gc.weight_min_value, gc.weight_max_value)
        for node, _, _, bias, response, links in reference.node_evals:
            gene = genome.nodes[node]
            if gene.activation not in ACTIVATIONS:
                raise ValueError(f"Activation '{gene.activation}' has no differentiable version")
            if gene.aggregation != "sum":
                raise ValueError("Only 'sum' aggregation is supported")

            bias_idx = add_param(bias, ("bias", node), bias_bounds)
            inputs = [(in_node, add_param(w, ("weight", (in_node, node)), weight_bounds))
                      for in_node, w in links]
            self.plan.append((node, ACTIVATIONS[gene.activation], bias_idx, response, inputs))

        self.theta0 = np.array(theta, dtype=float)    # birth (innate) parameters
        self.lo, self.hi = np.array(lo), np.array(hi)

    # ---------- forward pass ----------
    def forward(self, theta, X):
        """Network output for every row of X, on the scaled (-1..1) target scale."""
        n = X.shape[0]
        values = {k: X[:, i] for i, k in enumerate(self.input_keys)}
        for node, act, bias_idx, response, inputs in self.plan:
            s = sum((values[in_node] * theta[w_idx] for in_node, w_idx in inputs), anp.zeros(n))
            values[node] = act(theta[bias_idx] + response * s)
        # An output with no path from the inputs stays 0, as in neat-python
        return values.get(self.output_key, anp.zeros(n))

    def predict(self, theta, X):
        """Predictions in original target units."""
        return np.asarray(self.forward(theta, X)) * Y_SCALE + Y_CENTRE

    def mae(self, theta, data):
        """Mean absolute error on (X, y) in original target units."""
        X, y = data
        return float(np.mean(np.abs(self.predict(theta, X) - y)))

    # ---------- lifetime learning ----------
    def learn(self, data, steps=20, lr=0.05):
        """Run Adam on the training data and return the learned theta.

        The genome itself is NOT changed: the caller decides whether the learned
        parameters are thrown away (Baldwinian) or written back (Lamarckian).
        """
        X, y = data
        y_scaled = (y - Y_CENTRE) / Y_SCALE
        theta = self.theta0.copy()
        if theta.size == 0 or steps == 0:
            return theta

        loss_grad = grad(lambda t: anp.mean((self.forward(t, X) - y_scaled) ** 2))
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        for t in range(1, steps + 1):
            g = loss_grad(theta)
            m = BETA1 * m + (1 - BETA1) * g
            v = BETA2 * v + (1 - BETA2) * g ** 2
            m_hat, v_hat = m / (1 - BETA1 ** t), v / (1 - BETA2 ** t)
            # Clip so parameters stay inside NEAT's allowed ranges
            theta = np.clip(theta - lr * m_hat / (np.sqrt(v_hat) + EPS), self.lo, self.hi)
        return theta

    # ---------- write-back (for Lamarckian inheritance) ----------
    def write_to_genome(self, theta, genome):
        """Copy parameters from theta into the genome's weights and biases."""
        for (kind, key), value in zip(self.param_keys, theta):
            if kind == "bias":
                genome.nodes[key].bias = float(value)
            else:
                genome.connections[key].weight = float(value)
