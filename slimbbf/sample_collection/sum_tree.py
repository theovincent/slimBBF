# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/sum_tree.py

from functools import partial
import jax
import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.depth = int(np.ceil(np.log2(capacity))) + 1

        self.first_leaf_offset = (2 ** (self.depth - 1)) - 1
        self.nodes = np.zeros((2**self.depth) - 1, dtype=np.float64)
        self.max_recorded_priority = 1.0

    def set(self, indices, values):
        indices = np.atleast_1d(indices).astype(np.int32)
        values = np.atleast_1d(values).astype(np.float64)
        self.max_recorded_priority = max(self.max_recorded_priority, max(values))
        node_indices = self.first_leaf_offset + indices
        delta_values = values - self.nodes[node_indices]

        # De-duplicate indices, otherwise this can result in delta_values being
        # applied multiple times to the same index
        node_indices, unique_idx = np.unique(node_indices, return_index=True)
        delta_values = delta_values[unique_idx]
        for _ in reversed(range(self.depth - 1)):
            np.add.at(self.nodes, node_indices, delta_values)
            node_indices = (node_indices - 1) // 2

        np.add.at(self.nodes, node_indices, delta_values)

    def get(self, index):
        return self.nodes[self.first_leaf_offset + index]

    @property
    def root(self) -> float:
        return self.nodes[0]

    def query(self, targets):
        # Finds the smallest index where target < cumulative value up to index
        # We'll traverse the tree for all indices at once using masking
        node_indices = np.zeros_like(targets, dtype=np.int32)
        while (traversal_mask := node_indices < self.first_leaf_offset).any():
            left_node_indices = 2 * node_indices + 1
            left_node_sums = self.nodes[left_node_indices]
            right_node_indices = left_node_indices + 1

            # Traverse the tree but only if that index isn't masked
            node_indices = np.where(
                traversal_mask, np.where(targets < left_node_sums, left_node_indices, right_node_indices), node_indices
            )
            targets = np.where(targets < left_node_sums, targets, targets - left_node_sums)

        return node_indices - self.first_leaf_offset

    def stratified_sample(self, batch_size, rng):

        @partial(jax.jit, backend="cpu")
        def step(i, args):
            query_value, index, nodes = args
            left_child = index * 2 + 1
            left_sum = nodes[left_child]
            index = jax.lax.cond(query_value < left_sum, lambda x: x, lambda x: x + 1, left_child)
            query_value = jax.lax.cond(query_value < left_sum, lambda x: x, lambda x: x - left_sum, query_value)
            return query_value, index, nodes

        @partial(jax.jit, backend="cpu")
        @partial(jax.vmap, in_axes=(None, None, 0, None, None))
        def parallel_stratified_sample(rng, nodes, i, n, depth):
            rng = jax.random.fold_in(rng, i)
            total_priority = nodes[0]
            upper_bound = (i + 1) / n
            lower_bound = i / n
            query = jax.random.uniform(rng, minval=lower_bound, maxval=upper_bound)
            _, index, _ = jax.lax.fori_loop(0, depth, step, (query * total_priority, 0, nodes))
            return index

        indices = parallel_stratified_sample(rng, self.nodes, np.arange(batch_size), batch_size, self.depth)
        return np.minimum(indices - self.low_idx, self.highest_set)
