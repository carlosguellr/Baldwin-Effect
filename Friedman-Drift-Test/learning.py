"""
Lifetime learning for NEAT genomes: a differentiable copy of a NEAT network
plus Adam gradient descent.

neat.nn.FeedForwardNetwork evaluates one sample at a time in plain Python and
cannot compute gradients. GenomeNet builds the same network but evaluates all samples at once with NumPy.
It iss written with autograd.numpy, so autograd can differentiate the loss with respect to every weight and bias

All parameters (enabled connection weights + node biases) live in one flat
vector theta.

learning : Adam, full batch, on the training set, minimising MSE on the scaled target.
           Parameters are clipped to the genome's allowed range after every step,
           as NEAT mutation would.
"""
import autograd.numpy as anp
import numpy as np
from autograd import grad
from neat.nn import FeedForwardNetwork

from friedman_data import Y_CENTRE, Y_SCALE


# activation functions, made to match neat ones exactly. 
# written with anp rather than np so autograd can differentiate them
ACTIVATIONS = {
    "sigmoid":  lambda z: 1.0 / (1.0 + anp.exp(-anp.clip(5.0 * z, -60.0, 60.0))),
    "tanh":     lambda z: anp.tanh(anp.clip(2.5 * z, -60.0, 60.0)),
    "sin":      lambda z: anp.sin(anp.clip(5.0 * z, -60.0, 60.0)),
    "gauss":    lambda z: anp.exp(-5.0 * anp.clip(z, -3.4, 3.4) ** 2),
    "relu":     lambda z: anp.maximum(z, 0.0),
    "identity": lambda z: z,
}

# Adam hyperparameters
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


class GenomeNet:
    """Vectorised, differentiable version of a NEAT feed-forward network."""

    def __init__(self, genome, config):
        gc = config.genome_config # neat configuration
        self.input_keys = gc.input_keys
        self.output_key = gc.output_keys[0]           # single output

        # Build neat pythons network from the genome. 
        # never used for prediction but used as reference to ensure GenomeNet is built the same way
        reference = FeedForwardNetwork.create(genome, config)

        # stores the parameters, their bounds and what it is 
        theta, lo, hi = [], [], []
        self.param_keys = []    # per theta entry: ("bias", node) or ("weight", (in_node, node))
        # planfor forward pass: for each node what activation to use, which theta entries are bias and weights, and which nodes feed into it
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

        # Which inputs have an enabled path to the output. The plan only holds nodes
        # the output depends on, so any input feeding a planned node reaches the output.
        used = {in_node for *_, inputs in self.plan for in_node, _ in inputs}
        self.input_mask = np.array([k in used for k in self.input_keys])
        self.lo, self.hi = np.array(lo), np.array(hi)

    # forward pass 
    def forward(self, theta, X):
        """Run the network on every row of X at once (scaled target range)."""
        n_samples = X.shape[0]

        # outputs[node_id] = that node's output for every sample.
        # Input nodes output their column of X.
        outputs = {}
        for column, input_id in enumerate(self.input_keys):
            outputs[input_id] = X[:, column]

        # Compute the remaining nodes in order. Each node's inputs are already in `outputs`.
        for node_id, activation, bias_idx, response, incoming in self.plan:
            total = anp.zeros(n_samples)
            for source_id, weight_idx in incoming:
                total = total + outputs[source_id] * theta[weight_idx]
            outputs[node_id] = activation(theta[bias_idx] + response * total)

        # If nothing connects to the output node, the network predicts 0
        if self.output_key not in outputs:
            return anp.zeros(n_samples)
        return outputs[self.output_key]

    def predict(self, theta, X):
        """Predictions scaled to original target units."""
        return np.asarray(self.forward(theta, X)) * Y_SCALE + Y_CENTRE

    def mae(self, theta, data):
        """Mean absolute error on (X, y) in original target units."""
        X, y = data
        return float(np.mean(np.abs(self.predict(theta, X) - y)))

    def loss(self, theta, X, y_scaled):
        """MSE of the network on (X, y_scaled)."""
        predictions = self.forward(theta, X)
        return anp.mean((predictions - y_scaled) ** 2)

    def learn(self, data, steps=20, lr=0.05):
        X, y = data
        y_scaled = (y - Y_CENTRE) / Y_SCALE 
        theta = self.theta0.copy()               # start from the birth weights (copy because theta0 used for darwinian eval)

        # Nothing to learn: no parameters, or no learning steps
        if theta.size == 0 or steps == 0:
            return theta

        # gradient_of_loss(theta, X, y_scaled) returns one slope per parameter
        gradient_of_loss = grad(self.loss)

        # Adam's running averages, one value per parameter
        avg_gradient = np.zeros_like(theta)           # average direction (momentum)
        avg_squared_gradient = np.zeros_like(theta)   # average size, sets each parameter's step size

        for step in range(1, steps + 1):
            gradient = gradient_of_loss(theta, X, y_scaled)

            avg_gradient = BETA1 * avg_gradient + (1 - BETA1) * gradient
            avg_squared_gradient = BETA2 * avg_squared_gradient + (1 - BETA2) * gradient ** 2

            avg_gradient_corrected = avg_gradient / (1 - BETA1 ** step)
            avg_squared_corrected = avg_squared_gradient / (1 - BETA2 ** step)

            theta = theta - lr * avg_gradient_corrected / (np.sqrt(avg_squared_corrected) + EPS)
            theta = np.clip(theta, self.lo, self.hi)

        return theta

    # lamarckian inheritance
    def write_to_genome(self, theta, genome):
        """Copy parameters from theta into the genome's weights and biases."""
        for (kind, key), value in zip(self.param_keys, theta):
            if kind == "bias":
                genome.nodes[key].bias = float(value)
            else:
                genome.connections[key].weight = float(value)
