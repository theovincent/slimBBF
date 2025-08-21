# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/samplers.py
"""Sampling distributions."""

import numpy as np

from slimbbf.sample_collection import sum_tree


class UniformSamplingDistribution:
    """A uniform sampling distribution."""

    def __init__(self, seed: int) -> None:
        self._rng_key = np.random.default_rng(seed)
        self.index_max = -1

    def add(self, index):
        self.index_max = max(index, self.index_max)

    def sample(self, size):
        return self._rng_key.integers(0, self.index_max, size=size, endpoint=True)

    def get_probabilities(self, indices):
        return None


class PrioritizedSamplingDistribution(UniformSamplingDistribution):
    """A prioritized sampling distribution."""

    def __init__(self, seed: int, max_capacity: int) -> None:
        self._max_capacity = max_capacity
        self._sum_tree = sum_tree.SumTree(self._max_capacity)
        super().__init__(seed=seed)

    def add(self, index) -> None:
        super().add(index)
        self._sum_tree.set(index, self._sum_tree.max_recorded_priority)

    def update(self, metadata) -> None:
        priorities = np.where(metadata["loss"] == 0.0, 0.0, np.sqrt(metadata["loss"]))  # loss should be absolute
        self._sum_tree.set(metadata["indices"], priorities)

    def sample(self, size):
        if self._sum_tree.root == 0.0:
            return super().sample(size)
        targets = self._rng_key.uniform(0.0, self._sum_tree.root, size=size)
        return self._sum_tree.query(targets)

    def get_probabilities(self, indices):
        if self._sum_tree.root == 0.0:
            return np.ones_like(indices) / len(indices)
        return self._sum_tree.get(indices) / self._sum_tree.root
