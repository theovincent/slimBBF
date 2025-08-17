# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/replay_buffer.py
"""Simpler implementation of the subsequence replay memory (SPR style)."""
import numpy as np
from flax import struct

from slimdqn.sample_collection.replay_buffer import *


class SubsequenceReplayElement(struct.PyTreeNode):
    states_stack: np.ndarray[np.float64]
    actions_stack: np.ndarray[np.uint]
    reward: np.float32
    next_state: np.ndarray[np.float64]
    is_terminal: bool
    same_trajectory_mask: np.ndarray[bool]


class SubsequenceReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        sampling_distribution,
        max_capacity: int,
        batch_size: int,
        observation_shape: tuple,
        observation_dtype,
        spr_window: int = 5,
        stack_size: int = 4,
        update_horizon: int = 1,
        gamma: float = 0.99,
        clipping: callable = None,
        max_sample_trials=1000,
    ):
        super().__init__(
            sampling_distribution,
            max_capacity,
            batch_size,
            observation_shape,
            observation_dtype,
            stack_size,
            update_horizon,
            gamma,
            clipping,
            max_sample_trials,
        )
        self._spr_window = spr_window

    def _construct_batch_sample(self, index, first_terminal_index, n, gamma):
        states_stack = self._observation_stack[
            index_range(index - self._stack_size + 1, index + self._spr_window, self._max_capacity)
        ]
        states_stack = np.lib.stride_tricks.sliding_window_view(states_stack, window_shape=self._stack_size, axis=0)
        actions_stack = self._action_stack[index_range(index, index + self._spr_window, self._max_capacity)]
        is_terminal = first_terminal_index is not None
        reward = self._reward_stack[
            index_range(index, first_terminal_index if is_terminal else index + n - 1, self._max_capacity)
        ]

        reward = np.dot(reward, np.power(gamma, np.arange(len(reward))))
        next_state = np.moveaxis(
            self._observation_stack[index_range(index + n - self._stack_size + 1, index + n, self._max_capacity)], 0, -1
        )
        same_trajectory_mask = np.logical_and(
            (1 - self._is_terminal_stack[index_range(index, index + self._spr_window, self._max_capacity)]).cumprod(),
            (1 - self._is_truncation_stack[index_range(index, index + self._spr_window, self._max_capacity)]).cumprod(),
        )

        return SubsequenceReplayElement(
            states_stack, actions_stack, reward, next_state, is_terminal, same_trajectory_mask
        )
