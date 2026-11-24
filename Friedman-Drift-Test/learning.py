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

        theta, lo, hi = [], [], []
        self.param_keys = []    # per theta entry: ("bias", node_id) or ("weight", (source_id, node_id))
        self.plan = []          # per node: (node_id, activation, bias index, response, [(source_id, weight index)])

        def add_param(value, key, bounds):
            """Append one parameter to theta and return its index."""
            theta.append(value)
            self.param_keys.append(key)
            lo.append(bounds[0])
            hi.append(bounds[1])
            return len(theta) - 1

        bias_bounds = (gc.bias_min_value, gc.bias_max_value)
        weight_bounds = (gc.weight_min_value, gc.weight_max_value)
        for node_id, _, _, bias, response, links in reference.node_evals:
            gene = genome.nodes[node_id]
            if gene.activation not in ACTIVATIONS:
                raise ValueError(f"Activation '{gene.activation}' has no differentiable version")
            if gene.aggregation != "sum":
                raise ValueError("Only 'sum' aggregation is supported")

            bias_idx = add_param(bias, ("bias", node_id), bias_bounds)

            # Each incoming connection: (node it comes from, position of its weight in theta)
            incoming = []
            for source_id, weight in links:
                weight_idx = add_param(weight, ("weight", (source_id, node_id)), weight_bounds)
                incoming.append((source_id, weight_idx))

            self.plan.append((node_id, ACTIVATIONS[gene.activation], bias_idx, response, incoming))

        self.theta0 = np.array(theta, dtype=float)    # birth (innate) parameters

        # Which inputs have an enabled path to the output. The plan only holds nodes
        # the output depends on, so any input feeding a planned node reaches the output.
        used = set()
        for _, _, _, _, incoming in self.plan:
            for source_id, _ in incoming:
                used.add(source_id)
        self.input_mask = np.array([k in used for k in self.input_keys])
        self.lo, self.hi = np.array(lo), np.array(hi)

    # ---------- forward pass ----------
    def forward(self, theta, X):
        """Run the network on every row of X at once (scaled -1..1 target range)."""
        n_samples = X.shape[0]

        # outputs[node_id] = that node's output for every sample.
        # Input nodes simply output their column of X.
        outputs = {}
        for column, input_id in enumerate(self.input_keys):
            outputs[input_id] = X[:, column]

        # Compute the remaining nodes in order. Each node's inputs are already in `outputs`.
        for node_id, activation, bias_idx, response, incoming in self.plan:
            total = anp.zeros(n_samples)
            for source_id, weight_idx in incoming:
                total = total + outputs[source_id] * theta[weight_idx]
            outputs[node_id] = activation(theta[bias_idx] + response * total)

        # If nothing connects to the output node, the network predicts 0 (as in neat-python)
        if self.output_key not in outputs:
            return anp.zeros(n_samples)
        return outputs[self.output_key]

    def predict(self, theta, X):
        """Predictions in original target units."""
        return np.asarray(self.forward(theta, X)) * Y_SCALE + Y_CENTRE

    def mae(self, theta, data):
        """Mean absolute error on (X, y) in original target units."""
        X, y = data
        return float(np.mean(np.abs(self.predict(theta, X) - y)))

    # ---------- lifetime learning ----------
    def loss(self, theta, X, y_scaled):
        """Mean squared error of the network on (X, y_scaled). This is what learning minimises."""
        predictions = self.forward(theta, X)
        return anp.mean((predictions - y_scaled) ** 2)

    def learn(self, data, steps=20, lr=0.05):
        """Run Adam on the training data and return the learned theta.

        The genome itself is NOT changed: the caller decides whether the learned
        parameters are thrown away (Baldwinian) or written back (Lamarckian).
        """
        X, y = data
        y_scaled = (y - Y_CENTRE) / Y_SCALE      # train on the same -1..1 scale the network outputs
        theta = self.theta0.copy()               # start from the birth weights

        # Nothing to learn: no parameters, or no learning steps
        if theta.size == 0 or steps == 0:
            return theta

        # gradient_of_loss(theta, X, y_scaled) returns one slope per parameter:
        # how much the loss changes if that parameter is nudged
        gradient_of_loss = grad(self.loss)

        # Adam's running averages, one value per parameter
        avg_gradient = np.zeros_like(theta)           # average direction (momentum)
        avg_squared_gradient = np.zeros_like(theta)   # average size, sets each parameter's step size

        for step in range(1, steps + 1):
            gradient = gradient_of_loss(theta, X, y_scaled)

            # Update the running averages with this step's gradient
            avg_gradient = BETA1 * avg_gradient + (1 - BETA1) * gradient
            avg_squared_gradient = BETA2 * avg_squared_gradient + (1 - BETA2) * gradient ** 2

            # Both averages start at 0, so they are too small in early steps; this corrects that
            avg_gradient_corrected = avg_gradient / (1 - BETA1 ** step)
            avg_squared_corrected = avg_squared_gradient / (1 - BETA2 ** step)

            # Step downhill: against the average direction, scaled per parameter
            theta = theta - lr * avg_gradient_corrected / (np.sqrt(avg_squared_corrected) + EPS)

            # Keep parameters inside NEAT's allowed ranges, as mutation would
            theta = np.clip(theta, self.lo, self.hi)

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
