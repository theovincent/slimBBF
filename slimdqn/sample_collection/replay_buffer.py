# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/replay_buffer.py
"""Simpler implementation of the standard DQN replay memory."""
import collections
import operator

import jax
import numpy as np

from slimdqn.sample_collection.samplers import UniformSamplingDistribution, PrioritizedSamplingDistribution


def mod(x: int, N: int):
    return x % N


def index_range(a: int, b: int, N: int):
    a = a % N
    b = b % N
    if a <= b:
        return np.arange(a, b + 1)
    import pdb

    pdb.set_trace()
    return np.concat([np.arange(a, N), np.arange(b + 1)])


def compute_first_true_index(array, indices):
    all_indices = np.where(array[indices] > 0)[0]
    return all_indices[0] if len(all_indices) > 0 else indices[-1] + 1


class ReplayElement:
    def __init__(self, batch_size, state_shape):
        self.state = np.zeros((batch_size,) + state_shape, dtype=np.float32)
        self.action = np.zeros((batch_size,), dtype=np.int32)
        self.reward = np.zeros((batch_size,), dtype=np.float32)
        self.next_state = np.zeros((batch_size,) + state_shape, dtype=np.float32)
        self.is_terminal = np.zeros((batch_size,), dtype=np.int8)

    def update(self, index, state, action, reward, next_state, is_terminal):
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.next_state[index] = next_state
        self.is_terminal[index] = is_terminal


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
        self._highest_update_horizon = update_horizon  # set to highest that will be used in sampling
        self._gamma = gamma
        self._gamma_multipliers = np.power(gamma, np.arange(update_horizon))
        self._clipping = clipping
        self._max_sample_trials = max_sample_trials

    def add(self, observation, action, reward, is_terminal, is_truncation) -> None:
        self._observation_stack[mod(self.add_count, self._max_capacity)] = observation
        self._action_stack[mod(self.add_count, self._max_capacity)] = action
        self._reward_stack[mod(self.add_count, self._max_capacity)] = reward
        self._is_terminal_stack[mod(self.add_count, self._max_capacity)] = is_terminal
        self._is_truncation_stack[mod(self.add_count, self._max_capacity)] = is_truncation
        # THINK Prioritized RB add here --> how does it change?
        self.add_count += 1

    def sample(self, size=None, n=None):
        if size is None:
            size = self._batch_size
        if n is None:
            n = self._highest_update_horizon

        n_sample_trials = 0
        n_added_replay_elements = 0
        batch = ReplayElement(size, self._observation_shape + (self._stack_size,))
        while n_added_replay_elements < size and n_sample_trials < self._max_sample_trials:
            index = self._sampling_distribution.sample(0, min(self.add_count - 1, self._max_capacity - 1))
            is_valid, sample = self._check_valid(index, n)
            if is_valid:
                state, action, reward, next_state, is_terminal = sample
                batch.update(index, state, action, reward, next_state, is_terminal)
                n_added_replay_elements += 1
            n_sample_trials += 1
        assert n_added_replay_elements == size, "Could not construct a valid batch"
        return batch

    def update(self, keys, **kwargs):
        self._sampling_distribution.update(keys, **kwargs)

    def _check_valid(self, index, n):
        is_state_invalid = np.any(
            self._is_terminal_stack[index_range(index - self._stack_size + 1, index - 1, self._max_capacity)]
        ) or np.any(self._is_truncation_stack[index_range(index - self._stack_size + 1, index, self._max_capacity)])
        import pdb

        pdb.set_trace()

        first_terminal_index = mod(
            compute_first_true_index(self._is_terminal_stack, index_range(index, index + n - 1, self._max_capacity)),
            self._max_capacity,
        )
        first_truncation_index = mod(
            compute_first_true_index(self._is_truncation_stack, index_range(index, index + n - 1, self._max_capacity)),
            self._max_capacity,
        )
        is_next_state_valid = first_terminal_index <= first_truncation_index
        import pdb

        pdb.set_trace()

        if not is_state_invalid and is_next_state_valid:
            return True, self._construct_batch_sample(index, first_terminal_index, n)
        return False, None

    def _construct_batch_sample(self, index, first_terminal_index, n):
        state = self._observation_stack[index_range(index - self._stack_size + 1, index, self._max_capacity)]
        action = self._action_stack[index]
        reward = self._reward_stack[index_range(index, first_terminal_index, self._max_capacity)]
        reward = reward * self._gamma_multipliers[: len(reward)]
        next_state = self._observation_stack[
            index_range(index + n - self._stack_size + 1, index + n, self._max_capacity)
        ]
        is_terminal = first_terminal_index != mod(index + n, self._max_capacity)
        return (state, action, reward, next_state, is_terminal)
