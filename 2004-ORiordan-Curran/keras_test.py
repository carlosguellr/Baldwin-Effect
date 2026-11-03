"""
WHAT IT DOES
    Part 1  train N agents on both implementations and compare their outputs
            and fitness, and time each implementation per agent.

    Part 2 run a few generations of the real GA twice (same seed), once with
            the numpy evaluator and once with the Keras evaluator

    Part 3  use the measured per-agent times to estimate the cost of the full
            experiment (200 agents x 300 generations x 20 repeats x 2 learning experiments).

KERAS MODEL
    - One Dense(1, sigmoid) layer per decoded node, wired only to that node's sources, so the
      sparse topology is identical. 
    - Plain SGD, full batch (batch_size = 16 patterns), no shuffling, no early stopping:
      one epoch = one gradient step = one of our "training cycles".
    - Keras's 'mse' loss is mean((out-t)^2); numpy one uses the gradient of 0.5*mean((out-t)^2).
      The gradient therefore differs by a factor of 2, so Keras gets its learning_rate / 2.
    - Fitness is the fraction of patterns where |output - target| < 0.5 - MARGIN (MARGIN = 1e-6).
      Without the margin, an agent trained into a constant output of 0.5 +/- 1e-11 sits exactly
      on the boundary and float32 (Keras) vs float64 (numpy) rounding decides its score.
      
    - Edge cases follow network.py: no output node -> constant 0.5 output (fitness 0);
      a node with no usable inputs -> constant sigmoid(-threshold); duplicate connections
      from the same source -> the last weight wins.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "jax")      # must be set before keras is imported
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import time
import numpy as np
import keras

import experiment as E
from genome import decode, N_IN, OUT
from network import Network, raw_fitness

MARGIN = 1e-6    # decision margin, see the module docstring


def decision_fitness(out, t, margin=MARGIN):
    """Fraction of the 16 patterns classified on the correct side of 0.5 (decision-based fitness)."""
    return float(np.mean(np.abs(out - t) < 0.5 - margin))



def build_keras_model(nodes):
    """Build a Keras model with the same sparse wiring as Network(nodes)
        nodes is the dict returned by genome.decode(). 
    Returns None if there is no output node."""
    if OUT not in nodes:
        return None

    inputs = [keras.Input(shape=(1,)) for _ in range(N_IN)]   # one scalar input per pattern bit
    tensors = {i: inputs[i] for i in range(N_IN)}              # label -> that node's output tensor
    layers = {}                                                # label -> (layer, kernel, bias)

    # Nodes are built from low to high label
    for label in sorted(nodes):
        threshold, conns = nodes[label]

        # Same filtering as Network.__init__
        # ignore sources that are neither inputs nor defined nodes. 
        # A repeated source overwrites the earlier weight.
        weights = {}
        for src, w in conns:
            if src < N_IN or src in nodes:
                weights[src] = w

        if weights:
            srcs = list(weights)
            x = (tensors[srcs[0]] if len(srcs) == 1
                 else keras.layers.Concatenate()([tensors[s] for s in srcs]))
            kernel = np.array([[weights[s]] for s in srcs], dtype="float32")
        else:
            # No usable inputs: our numpy node outputs sigmoid(-threshold) for every pattern.
            x = keras.layers.Lambda(lambda z: z * 0.0)(inputs[0])
            kernel = np.zeros((1, 1), dtype="float32")

        dense = keras.layers.Dense(1, activation="sigmoid")
        tensors[label] = dense(x)
        layers[label] = (dense, kernel, np.array([-threshold], dtype="float32"))

    model = keras.Model(inputs, tensors[OUT])
    for dense, kernel, bias in layers.values():                # load the genome's weights
        dense.set_weights([kernel, bias])
    return model


def evaluate_keras(genome, X, t, cycles, learning_rate):
    """Keras counterpart of network.evaluate(): returns (fitness, outputs).
    Includes everything an agent costs: decode, build, compile, train, test."""
    nodes = decode(genome)
    model = build_keras_model(nodes)
    if model is None:                                          # no output node: constant 0.5
        out = np.full(len(X), 0.5)
        return decision_fitness(out, t), out

    cols = [X[:, i:i + 1] for i in range(N_IN)]                # Keras requires one array per Input
    if cycles > 0:
        # learning_rate / 2 compensates for Keras's mse having a factor 2
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate / 2), loss="mse")
        model.fit(cols, t, epochs=cycles, batch_size=len(X), shuffle=False, verbose=0)
    out = np.asarray(model(cols)).ravel()
    keras.backend.clear_session()                              # free the graph; avoids memory build-up
    return decision_fitness(out, t), out


def numpy_outputs(genome, X, t, cycles, learning_rate):
    """numpy implementation (network.py): decode, build, train, test. 
        Returns (fitness, outputs)."""
    net = Network(decode(genome))
    if cycles > 0:
        net.train(X, t, cycles, learning_rate)
    out = net.output(X)
    return decision_fitness(out, t), out


#test 1 - comparing both implementations on the same genomes, checking outputs and fitness
def test1_comparison(n_agents, cycles, lr, seed):
    """Train the same genomes on both implementations; compare outputs, fitness and speed."""
    rng = np.random.default_rng(seed)
    env = E.Environment()
    # Half structured genomes half random-bit genomes
    genomes = [E.structured_genome(rng) if i % 2 == 0 else E.random_bits_genome(rng)
               for i in range(n_agents)]

    #the first Keras call pays one-off start-up costs which is excluded from the timings.
    evaluate_keras(E.structured_genome(rng), env.X, env.t, cycles, lr)

    diffs, fit_mismatch, with_output, boundary, strict_mismatch = [], 0, 0, 0, 0
    t_np = t_keras = 0.0
    for g in genomes:
        t0 = time.perf_counter()
        f_np, out_np = numpy_outputs(g, env.X, env.t, cycles, lr)
        t_np += time.perf_counter() - t0

        t0 = time.perf_counter()
        f_k, out_k = evaluate_keras(g, env.X, env.t, cycles, lr)
        t_keras += time.perf_counter() - t0

        with_output += OUT in decode(g)
        diffs.append(np.abs(out_np - out_k).max())             # worst pattern for this agent
        fit_mismatch += (f_np != f_k)
        # agents with an output within 1e-6 of the decision boundary, and how often a strict
        # "< 0.5" rule (no margin) would have scored the two implementations differently
        boundary += bool(np.any(np.abs(np.abs(out_np - env.t) - 0.5) < 1e-6))
        strict_mismatch += (np.mean(np.abs(out_np - env.t) < 0.5) != np.mean(np.abs(out_k - env.t) < 0.5))
    return dict(boundary=boundary, strict_mismatch=strict_mismatch, n=n_agents, with_output=with_output, max_diff=max(diffs), median_diff=float(np.median(diffs)),
                fit_mismatch=fit_mismatch, ms_np=1000 * t_np / n_agents, ms_keras=1000 * t_keras / n_agents)



# test 2 - subset of the real experiment with the same seed. 
def mini_experiment(evaluator, pop_size, generations, flip_at, cycles, lr, seed):
    """A short GA run with a sudden flip at flip_at.
    The GA's randomness comes only from seed, so two evaluators that give identical fitness values produce identical runs."""
    rng = np.random.default_rng(seed)
    env = E.Environment()
    pop = [E.structured_genome(rng) for _ in range(pop_size)]
    means, bests = [], []
    t0 = time.perf_counter()
    for gen in range(generations):
        if gen == flip_at:
            env.flip_all()                                      # sudden environment change
        fit = np.array([evaluator(g, env.X, env.t, cycles, lr) for g in pop])
        means.append(fit.mean())
        bests.append(fit.max())
        pop = E.next_generation(pop, E.linear_rank(fit), rng)   # rank, select, crossover, mutate
    return np.array(means), np.array(bests), time.perf_counter() - t0


if __name__ == "__main__":
    AGENTS = 30      # agents in the first test
    POP = 20         # population size of the mini experiment
    GENS = 6         # generations of the mini experiment
    CYCLES = 50      # training cycles per lifetime
    LR = 3.0         # learning rate
    CORES = 20       # cores assumed in the cost projection
    SEED = 1
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"Settings: {CYCLES} cycles per lifetime, learning rate {LR}, Keras backend: {keras.backend.backend()}")
 
    # Agreement test: the same genomes trained on both implementations.
    r = test1_comparison(AGENTS, CYCLES, LR, SEED)
    say("\nAgreement on identical genomes")
    say(f"  agents tested: {r['n']}")
    say(f"  largest output difference (any agent, any pattern): {r['max_diff']:.2e}   median over agents: {r['median_diff']:.2e}")
    say(f"  time per agent: numpy {r['ms_np']:.1f} ms | Keras {r['ms_keras']:.0f} ms  ->  Keras is {r['ms_keras'] / r['ms_np']:.0f}x slower")
    say()
 
    # Mini experiment: the real GA, run once per implementation with the same seed.
    flip = GENS // 2
    m_np, b_np, s_np = mini_experiment(lambda g, X, t, c, lr: numpy_outputs(g, X, t, c, lr)[0],
                                       POP, GENS, flip, CYCLES, LR, SEED)
    m_k, b_k, s_k = mini_experiment(lambda g, X, t, c, lr: evaluate_keras(g, X, t, c, lr)[0],
                                    POP, GENS, flip, CYCLES, LR, SEED)
    say(f"\nMini experiment ({POP} agents x {GENS} generations, sudden flip at generation {flip})")
    say("  generation      " + " ".join(f"{g:6d}" for g in range(GENS)))
    say("  mean  numpy     " + " ".join(f"{v:6.3f}" for v in m_np))
    say("  mean  Keras     " + " ".join(f"{v:6.3f}" for v in m_k))
    say("  best  numpy     " + " ".join(f"{v:6.3f}" for v in b_np))
    say("  best  Keras     " + " ".join(f"{v:6.3f}" for v in b_k))
    if np.allclose(m_np, m_k) and np.allclose(b_np, b_k):
        say("  identical fitness curves")
    else:
        say(f"  fitness curves differ (largest difference in the mean curve: {np.abs(m_np - m_k).max():.3f})")
    say(f"  wall time: numpy {s_np:.1f} s | Keras {s_k:.1f} s")
 
    # Projected cost of the full experiment: 200 agents x 300 generations per run, 40 learning runs.
    agents_per_run = E.POP_SIZE * E.GENERATIONS
    learning_runs = 2 * 20                       # gradual + sudden, 20 repeats each
    say("\nProjected cost of the full experiment")
    for name, ms in (("numpy", r["ms_np"]), ("Keras", r["ms_keras"])):
        hours_on_cores = agents_per_run * learning_runs * ms / 1000 / 3600 / CORES
        say(f"    {name}: {agents_per_run * ms / 1000 / 60:7.1f} min per run | about {hours_on_cores:5.1f} h on {CORES} cores")
    say("  (from the per-agent times above)")
 
    with open("keras_vs_numpy.txt", "w") as f:
        f.write("\n".join(lines) + "\n")