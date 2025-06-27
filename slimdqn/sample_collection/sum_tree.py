# Inspired from: https://github.com/google-research/google-research/blob/master/bigger_better_faster/bbf/replay_memory/deterministic_sum_tree.py

"""A sum tree data structure that uses JAX for controlling randomness."""

import functools

import jax
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, backend="cpu")
def step(i, args):
    query_value, index, nodes = args
    left_child = index * 2 + 1
    left_sum = nodes[left_child]
    index = jax.lax.cond(query_value < left_sum, lambda x: x, lambda x: x + 1, left_child)
    query_value = jax.lax.cond(query_value < left_sum, lambda x: x, lambda x: x - left_sum, query_value)
    return query_value, index, nodes


@functools.partial(jax.jit, backend="cpu")
@functools.partial(jax.vmap, in_axes=(None, None, 0, None, None))
def parallel_stratified_sample(rng, nodes, i, n, depth):
    rng = jax.random.fold_in(rng, i)
    total_priority = nodes[0]
    upper_bound = (i + 1) / n
    lower_bound = i / n
    query = jax.random.uniform(rng, minval=lower_bound, maxval=upper_bound)
    _, index, _ = jax.lax.fori_loop(0, depth, step, (query * total_priority, 0, nodes))
    return index


class DeterministicSumTree:
    """A sum tree data structure for storing replay priorities."""

    def __init__(self, capacity):
        assert capacity > 0, "Give non-negative capacity"
        self.nodes = []
        self.depth = int(np.ceil(np.log2(capacity)))
        self.low_idx = (2**self.depth) - 1  # node_idx + low_idx -> tree_idx
        self.high_idx = capacity + self.low_idx
        self.nodes = np.zeros(2 ** (self.depth + 1) - 1)
        self.capacity = capacity

        self.highest_set = 0

        self.max_recorded_priority = 1.0

    def _total_priority(self):
        """Returns the sum of all priorities stored in this sum tree."""
        return self.nodes[0]

    def query(self, query_value):
        """Samples an element from the sum tree based on query."""
        assert self._total_priority() > 0, "Can't query empty tree"
        nodes = jnp.array(self.nodes)
        query_value *= self._total_priority()

        _, index, _ = jax.lax.fori_loop(0, self.depth, step, (query_value, 0, nodes))

        return np.minimum(index - self.low_idx, self.highest_set)

    def stratified_sample(self, batch_size, rng):
        """Performs stratified sampling using the sum tree."""
        assert self._total_priority() > 0.0, "Cannot sample from an empty sum tree."
        indices = parallel_stratified_sample(rng, self.nodes, np.arange(batch_size), batch_size, self.depth)
        return np.minimum(indices - self.low_idx, self.highest_set)

    def get(self, node_index):
        """Returns the value of the leaf node corresponding to the index."""
        return self.nodes[node_index + self.low_idx]

    def reset_priorities(self):
        # CHECK
        for i in range(self.highest_set):
            self.set(i, self.max_recorded_priority)

    def set(self, node_index, value):
        """Sets the value of a leaf node and updates internal nodes accordingly."""
        assert value >= 0.0, "Sum tree values should be nonnegative. Got {}".format(value)
        self.highest_set = max(node_index, self.highest_set)
        node_index = node_index + self.low_idx
        self.max_recorded_priority = max(value, self.max_recorded_priority)

        delta_value = value - self.nodes[node_index]

        # Now traverse back the tree, adjusting all sums along the way.
        for _ in reversed(range(self.depth)):
            # Note: Adding a delta leads to some tolerable numerical inaccuracies.
            self.nodes[node_index] += delta_value
            node_index = (node_index - 1) // 2

        self.nodes[node_index] += delta_value
        assert node_index == 0, "Sum tree traversal failed, final node index is not 0."
