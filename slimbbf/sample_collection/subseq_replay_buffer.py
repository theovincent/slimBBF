# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/replay_buffer.py
import numpy as np
from flax import struct

from slimbbf.sample_collection.replay_buffer import ReplayBuffer, mod_index_range


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
        max_capacity: int,
        seed: int,
        batch_size: int,
        observation_shape: tuple,
        observation_dtype,
        spr_window: int,
        stack_size: int,
        clipping: callable,
    ):
        super().__init__(max_capacity, seed, batch_size, observation_shape, observation_dtype, stack_size, clipping)
        self._spr_window = spr_window

    def _construct_batch_sample(self, index, first_terminal_index, n, gamma):
        # if first_terminal_index == None, the sample is a regular sample
        # if first_terminal_index != None, the sample is terminal and the reward should be accumulated until first_terminal_index.
        # Get frames of state of shape (stack_size, H, W) from observation_stack and change to (H, W, stack_size)
        state_stack_index_range = mod_index_range(
            index - self._stack_size + 1, index + self._spr_window, self._max_capacity
        )
        states_stack = self._observation_stack[state_stack_index_range]
        states_stack = np.lib.stride_tricks.sliding_window_view(states_stack, window_shape=self._stack_size, axis=0)

        actions_stack = self._action_stack[mod_index_range(index, index + self._spr_window, self._max_capacity)]
        is_terminal = first_terminal_index is not None

        reward_terms = self._reward_stack[
            mod_index_range(index, first_terminal_index if is_terminal else index + n - 1, self._max_capacity)
        ]
        reward = np.dot(reward_terms, np.power(gamma, np.arange(len(reward_terms))))

        # Get frames of next state of shape (stack_size, H, W) from observation_stack and change to (H, W, stack_size)
        # if is_terminal, then the next state will be ignored so it is irrelevant
        next_state_index_range = mod_index_range(index + n - self._stack_size + 1, index + n, self._max_capacity)
        next_state = np.moveaxis(self._observation_stack[next_state_index_range], 0, -1)

        # Constructs mask for SPR loss with True for all indices in same trajectory as state
        spr_obs_indices = mod_index_range(index, index + self._spr_window, self._max_capacity)
        trajectory_end_flags = np.loigcal_or(
            self._is_terminal_stack[spr_obs_indices], self._is_truncation_stack[spr_obs_indices]
        )
        next_states_in_trajectory = (1 - trajectory_end_flags).cumprod()
        # Sets mask to True for terminal/truncating state if trajectory ends
        states_in_trajectory = np.concatenate(([True], next_states_in_trajectory[:-1]))

        return SubsequenceReplayElement(
            states_stack, actions_stack, reward, next_state, is_terminal, states_in_trajectory
        )
