# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/replay_buffer.py
"""Simpler implementation of the standard DQN replay memory."""
import jax
import numpy as np

from flax import struct

from slimdqn.sample_collection.samplers import UniformSamplingDistribution, PrioritizedSamplingDistribution


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
    action: int
    reward: float
    next_state: np.ndarray[np.float64]
    is_terminal: bool


class ReplayBuffer:

    def __init__(
        self,
        sampling_distribution: UniformSamplingDistribution,
        batch_size: int,
        max_capacity: int,
        observation_shape: tuple,
        stack_size: int = 4,
        update_horizon: int = 1,
        gamma: float = 0.99,
        clipping: callable = None,
        max_sample_trials=1000,
    ):
        self.add_count = 0
        self._max_capacity = max_capacity
        self._observation_shape = observation_shape
        self._observation_stack = np.zeros((self._max_capacity,) + observation_shape, dtype=np.float32)
        self._action_stack = np.zeros((self._max_capacity,), dtype=np.int32)
        self._reward_stack = np.zeros((self._max_capacity,), dtype=np.float32)
        self._is_terminal_stack = np.ones((self._max_capacity,), dtype=np.int8)
        self._is_truncation_stack = np.ones((self._max_capacity,), dtype=np.int8)

        self._sampling_distribution = sampling_distribution
        self._batch_size = batch_size

        self._stack_size = stack_size
        self._highest_update_horizon = update_horizon
        self._gamma = gamma
        self._clipping = clipping
        self._max_sample_trials = max_sample_trials

        self._last_is_truncation = False

    def add(self, observation, action, reward, is_terminal, is_truncation) -> None:
        if self.add_count == 0:
            self._observation_stack[
                index_range(self.add_count, self.add_count + self._stack_size - 1, self._max_capacity)
            ] = 0
            self._is_terminal_stack[
                index_range(self.add_count, self.add_count + self._stack_size - 1, self._max_capacity)
            ] = 0
            self._is_truncation_stack[
                index_range(self.add_count, self.add_count + self._stack_size - 1, self._max_capacity)
            ] = 0
            self.add_count += self._stack_size - 1

        self._observation_stack[mod(self.add_count, self._max_capacity)] = observation
        self._action_stack[mod(self.add_count, self._max_capacity)] = action
        self._reward_stack[mod(self.add_count, self._max_capacity)] = reward
        self._is_terminal_stack[mod(self.add_count, self._max_capacity)] = is_terminal
        self._is_truncation_stack[mod(self.add_count, self._max_capacity)] = True
        if self.add_count >= self._stack_size:
            self._is_truncation_stack[mod(self.add_count - 1, self._max_capacity)] = self._last_is_truncation
        self._last_is_truncation = is_truncation
        # THINK Prioritized RB add here --> how does it change?
        self.add_count += 1
        if is_terminal or is_truncation:
            self._observation_stack[
                index_range(self.add_count, self.add_count + self._stack_size - 1, self._max_capacity)
            ] = 0
            self._is_terminal_stack[
                index_range(self.add_count, self.add_count + self._stack_size - 1, self._max_capacity)
            ] = 0
            self._is_truncation_stack[
                index_range(self.add_count, self.add_count + self._stack_size - 1, self._max_capacity)
            ] = 0
            self.add_count += self._stack_size - 1

    def sample(self, batch_size=None, n=None, gamma=None):
        if batch_size is None:
            batch_size = self._batch_size
        if n is None:
            n = self._highest_update_horizon
        if gamma is None:
            gamma = self._gamma

        batch = []
        indices = self._sampling_distribution.sample(0, min(self.add_count - 1, self._max_capacity - 1), batch_size)
        for index in indices:
            n_sample_trials = 0
            is_valid, sample = self._check_valid(index, n, gamma)
            while (not is_valid) and n_sample_trials < self._max_sample_trials:
                index = self._sampling_distribution.sample(0, min(self.add_count - 1, self._max_capacity - 1), 1)[0]
                n_sample_trials += 1
                is_valid, sample = self._check_valid(index, n, gamma)

            assert is_valid, "Could not construct a valid batch"
            state, action, reward, next_state, is_terminal = sample
            batch.append(ReplayElement(state, action, reward, next_state, is_terminal))
        return jax.tree_util.tree_map(lambda *xs: np.stack(xs), *batch)

    def update(self, keys, **kwargs):
        self._sampling_distribution.update(keys, **kwargs)

    def _check_valid(self, index, n, gamma):
        is_state_invalid = self._stack_size > 1 and (
            np.any(self._is_terminal_stack[index_range(index - self._stack_size + 1, index - 1, self._max_capacity)])
            or np.any(self._is_truncation_stack[index_range(index - self._stack_size + 1, index, self._max_capacity)])
        )

        first_terminal_index = mod(
            compute_first_true_index(self._is_terminal_stack, index_range(index, index + n - 1, self._max_capacity)),
            self._max_capacity,
        )
        first_truncation_index = mod(
            compute_first_true_index(self._is_truncation_stack, index_range(index, index + n - 1, self._max_capacity)),
            self._max_capacity,
        )
        is_next_state_valid = first_terminal_index <= first_truncation_index

        if not is_state_invalid and is_next_state_valid:
            return True, self._construct_batch_sample(index, first_terminal_index, n, gamma)
        return False, None

    def _construct_batch_sample(self, index, first_terminal_index, n, gamma):

        state = self._observation_stack[index_range(index - self._stack_size + 1, index, self._max_capacity)].transpose(
            np.append(np.arange(1, len(self._observation_shape) + 1), 0)
        )
        action = self._action_stack[index]
        is_terminal = first_terminal_index != mod(index + n, self._max_capacity)
        reward = self._reward_stack[
            index_range(index, first_terminal_index if is_terminal else first_terminal_index - 1, self._max_capacity)
        ]

        reward = np.dot(reward, np.power(gamma, np.arange(len(reward))))
        next_state = self._observation_stack[
            index_range(index + n - self._stack_size + 1, index + n, self._max_capacity)
        ].transpose(np.append(np.arange(1, len(self._observation_shape) + 1), 0))

        return (state, action, reward, next_state, is_terminal)
