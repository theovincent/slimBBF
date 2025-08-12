# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/replay_buffer.py
"""Simpler implementation of the standard DQN replay memory (supports Uniform and PER sampler)."""
import jax
import numpy as np
import jax.numpy as jnp

from flax import struct


def mod(x: int, N: int):
    return x % N


def index_range(a: int, b: int, N: int):
    a = mod(a, N)
    b = mod(b, N)
    if a <= b:
        return np.arange(a, b + 1)
    return np.concat([np.arange(a, N), np.arange(b + 1)])


def compute_first_true_index(array, indices):
    all_indices = np.where(array[indices] > 0)[0]
    return indices[all_indices[0]] if len(all_indices) > 0 else indices[-1] + 1


class ReplayElement(struct.PyTreeNode):
    state: np.ndarray[np.float64]
    action: np.uint
    reward: np.float32
    next_state: np.ndarray[np.float64]
    is_terminal: bool


class ReplayBuffer:

    def __init__(
        self,
        sampling_distribution,
        max_capacity: int,
        batch_size: int,
        observation_shape: tuple,
        observation_dtype,
        stack_size: int = 4,
        update_horizon: int = 1,
        gamma: float = 0.99,
        clipping: callable = None,
        max_sample_trials=1000,
    ):
        assert max_capacity >= stack_size, "Need at least stack_size capacity in replay buffer"
        self._max_capacity = max_capacity
        self._observation_shape = observation_shape
        self._observation_stack = np.zeros((max_capacity,) + observation_shape, dtype=observation_dtype)
        self._action_stack = np.zeros((max_capacity,), dtype=np.uint32)
        self._reward_stack = np.zeros((max_capacity,), dtype=np.float32)
        self._is_terminal_stack = np.ones((max_capacity,), dtype=np.uint8)
        self._is_truncation_stack = np.ones((max_capacity,), dtype=np.uint8)
        self._is_terminal_stack[:stack_size] = 0
        self._is_truncation_stack[:stack_size] = 0
        self.add_count = stack_size - 1

        self._sampling_distribution = sampling_distribution
        self._batch_size = batch_size

        self._stack_size = stack_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._clipping = clipping
        self._max_sample_trials = max_sample_trials

        self._last_is_truncation = False

    def add(self, observation, action, reward, is_terminal, is_truncation) -> None:
        self._observation_stack[mod(self.add_count, self._max_capacity)] = observation
        self._action_stack[mod(self.add_count, self._max_capacity)] = action
        self._reward_stack[mod(self.add_count, self._max_capacity)] = reward
        self._is_terminal_stack[mod(self.add_count, self._max_capacity)] = is_terminal
        self._is_truncation_stack[mod(self.add_count, self._max_capacity)] = (
            False if is_terminal else True
        )  # we technically truncate if stop the run here
        if self.add_count >= self._stack_size:
            self._is_truncation_stack[mod(self.add_count - 1, self._max_capacity)] = (
                self._last_is_truncation
                if not self._is_terminal_stack[mod(self.add_count - 1, self._max_capacity)]
                else False  # truncation is False if terminated
            )
        self._last_is_truncation = is_truncation
        self._sampling_distribution.add(mod(self.add_count, self._max_capacity))
        self.add_count += 1
        if (is_terminal or is_truncation) and self._stack_size > 1:  # to fill zeroed frames
            self._observation_stack[
                index_range(self.add_count, self.add_count + self._stack_size - 2, self._max_capacity)
            ] = 0
            self._is_terminal_stack[
                index_range(self.add_count, self.add_count + self._stack_size - 2, self._max_capacity)
            ] = 0
            self._is_truncation_stack[
                index_range(self.add_count, self.add_count + self._stack_size - 2, self._max_capacity)
            ] = 0
            self.add_count += self._stack_size - 1

    def sample(self, batch_size=None, n=None, gamma=None):
        if batch_size is None:
            batch_size = self._batch_size
        if n is None:
            n = self._update_horizon
        if gamma is None:
            gamma = self._gamma

        batch = []
        batch_indices = []
        indices = self._sampling_distribution.sample(size=batch_size)
        for index in indices:
            n_sample_trials = 1
            sample = self._check_valid_and_get_sample(index, n, gamma)
            while (not sample) and n_sample_trials < self._max_sample_trials:
                index = self._sampling_distribution.sample(size=1)[0]
                n_sample_trials += 1
                sample = self._check_valid_and_get_sample(index, n, gamma)

            assert sample, "Could not construct a valid batch"
            batch_indices.append(index)
            batch.append(ReplayElement(*sample))

        return jax.tree_util.tree_map(lambda *xs: np.stack(xs), *batch), {
            "indices": jnp.array(batch_indices),
            "probabilities": self._sampling_distribution.get_probabilities(batch_indices),
        }

    def update(self, metadata):  # using with UniformSamplingDistribution gives error
        self._sampling_distribution.update(metadata)

    def _check_valid_and_get_sample(self, index, n, gamma):
        is_state_invalid = self._stack_size > 1 and (
            np.any(self._is_terminal_stack[index_range(index - self._stack_size + 1, index - 1, self._max_capacity)])
            or np.any(self._is_truncation_stack[index_range(index - self._stack_size + 1, index, self._max_capacity)])
        )

        index_range_for_rewards = index_range(index, index + n - 1, self._max_capacity)
        first_terminal_index = mod(
            compute_first_true_index(self._is_terminal_stack, index_range_for_rewards), self._max_capacity
        )
        first_truncation_index = mod(
            compute_first_true_index(self._is_truncation_stack, index_range_for_rewards), self._max_capacity
        )
        is_next_state_valid = (
            first_terminal_index in index_range_for_rewards or first_truncation_index not in index_range_for_rewards
        )

        if (not is_state_invalid) and is_next_state_valid:
            return self._construct_batch_sample(index, first_terminal_index, n, gamma)
        return None

    def _construct_batch_sample(self, index, first_terminal_index, n, gamma):
        state = np.moveaxis(
            self._observation_stack[index_range(index - self._stack_size + 1, index, self._max_capacity)], 0, -1
        )
        action = self._action_stack[index]
        is_terminal = first_terminal_index != mod(index + n, self._max_capacity)
        reward = self._reward_stack[index_range(index, first_terminal_index - (not is_terminal), self._max_capacity)]

        reward = np.dot(reward, np.power(gamma, np.arange(len(reward))))
        next_state = np.moveaxis(
            self._observation_stack[index_range(index + n - self._stack_size + 1, index + n, self._max_capacity)], 0, -1
        )

        return (state, action, reward, next_state, is_terminal)
