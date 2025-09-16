# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/replay_buffer.py
import jax
import numpy as np
from flax import struct

from slimbbf.sample_collection import sum_tree


def mod_index_range(i: int, j: int, N: int):
    # return the range [i, j] modulo N with j included
    i_mod = i % N
    j_mod = j % N
    return np.arange(i_mod, j_mod + 1) if i_mod <= j_mod else np.concat([np.arange(i_mod, N), np.arange(j_mod + 1)])


class ReplayElement(struct.PyTreeNode):
    state: np.ndarray[np.float64]
    action: np.uint
    reward: np.float32
    next_state: np.ndarray[np.float64]
    is_terminal: bool


class ReplayBuffer:
    def __init__(
        self,
        max_capacity: int,
        seed: int,
        batch_size: int,
        observation_shape: tuple,
        observation_dtype,
        stack_size: int,
        clipping: callable,
    ):
        self.max_capacity = max_capacity
        self.rng_key = np.random.default_rng(seed)

        self.observation_stack = np.zeros((max_capacity,) + observation_shape, dtype=observation_dtype)
        self.action_stack = np.zeros((max_capacity,), dtype=np.uint)
        self.reward_stack = np.zeros((max_capacity,), dtype=np.float32)
        self.is_terminal_stack = np.ones((max_capacity,), dtype=np.uint8)
        self.is_truncation_stack = np.ones((max_capacity,), dtype=np.uint8)

        self.batch_size = batch_size
        self.stack_size = stack_size
        self.clipping = clipping

        self.sum_tree = sum_tree.SumTree(max_capacity)

        # Fill initial zero frames
        self.is_terminal_stack[: stack_size - 1] = 0
        self.is_truncation_stack[: stack_size - 1] = 0
        self.add_count = stack_size - 1
        self.last_obs_is_truncation = True

    def add(self, observation, action, reward, is_terminal, is_truncation) -> None:
        add_index = self.add_count % self.max_capacity
        self.observation_stack[add_index] = observation
        self.action_stack[add_index] = action
        self.reward_stack[add_index] = reward
        self.is_terminal_stack[add_index] = is_terminal
        # Always truncate in case we are overwriting the buffer and
        # the next observation is not belonging to the same episode
        self.is_truncation_stack[add_index] = False if is_terminal else True

        # Update the truncation flag for the previous frame if trajectory did not truncate
        # If the trajectory ended, self.add_count - 1 will not correspond to the accurate index,
        # it will correspond to a black observation, which is not a truncated state so _is_truncation = False
        if not self.last_obs_is_truncation:
            self.is_truncation_stack[(self.add_count - 1) % self.max_capacity] = False
        self.last_obs_is_truncation = is_truncation

        self.add_count += 1

        # Fill zeroed frames in case of end of trajectory
        if (is_terminal or is_truncation) and self.stack_size > 1:
            zeroed_indices = mod_index_range(self.add_count, self.add_count + self.stack_size - 2, self.max_capacity)
            self.observation_stack[zeroed_indices] = 0
            self.action_stack[zeroed_indices] = 0
            self.reward_stack[zeroed_indices] = 0
            self.is_terminal_stack[zeroed_indices] = 0
            self.is_truncation_stack[zeroed_indices] = 0
            self.add_count += self.stack_size - 1

        self.sum_tree.set(add_index, self.sum_tree.max_recorded_priority)

    def sample(self, n, gamma, n_batches):

        effective_batch_size = n_batches * self.batch_size
        cdf_segments = np.linspace(0, self.sum_tree.root, effective_batch_size + 1)
        cdf_targets = cdf_segments[:-1] + self.rng_key.random(effective_batch_size) * (
            cdf_segments[1:] - cdf_segments[:-1]
        )
        initial_indices = self.sum_tree.query(cdf_targets)

        batches = []
        indices = []
        for idx_batch in range(n_batches):
            batch = []
            batch_indices = []
            for i, index in enumerate(initial_indices[idx_batch * self.batch_size : (idx_batch + 1) * self.batch_size]):
                n_sample_trials = 1
                sample = self.check_valid_and_get_sample(index, n, gamma, batch_zero_mask=idx_batch == 0 and i == 0)

                # Check if sample is not None until valid sample or 1000 trial limit
                while sample is None and n_sample_trials < 1000:
                    index = self.sum_tree.query(self.rng_key.uniform(0.0, self.sum_tree.root, size=1))[0]
                    n_sample_trials += 1
                    sample = self.check_valid_and_get_sample(index, n, gamma, batch_zero_mask=idx_batch == 0 and i == 0)

                assert sample, f"Could not construct a valid batch after {n_sample_trials} trials"

                batch_indices.append(index)
                batch.append(sample)

            batches.append(batch)
            indices.append(batch_indices)

        indices = np.array(indices)
        probabilities = self.sum_tree.get(indices.reshape(-1)).reshape((n_batches, self.batch_size))
        importance_weights = 1.0 / np.sqrt(probabilities + 1e-10)  # beta = 0.5
        importance_weights /= np.max(importance_weights)

        return (
            [jax.tree_util.tree_map(lambda *xs: np.stack(xs), *batch) for batch in batches],
            indices,
            importance_weights,
        )

    def check_valid_and_get_sample(self, index, n, gamma, **kwargs):
        # Is state valid?
        state_stack_indices_except_last = mod_index_range(index - self.stack_size + 1, index - 1, self.max_capacity)
        is_state_invalid = self.stack_size > 1 and (
            np.any(self.is_terminal_stack[state_stack_indices_except_last])
            or np.any(self.is_truncation_stack[state_stack_indices_except_last])
        )

        # Is next state valid?
        indices_upto_next_state = mod_index_range(index, index + n - 1, self.max_capacity)
        indices_for_is_terminal = mod_index_range(index, index + n - 2, self.max_capacity)
        # Find the first index in indices_upto_next_state where terminal/truncation is true
        # if does not exists set it to None
        terminal_indices_true = np.where(self.is_terminal_stack[indices_upto_next_state])[0]
        terminal_indices_one_less_n = np.where(self.is_terminal_stack[indices_for_is_terminal])[0]

        first_terminal_index = (
            indices_upto_next_state[terminal_indices_true[0]] if len(terminal_indices_true) > 0 else None
        )
        is_terminal_for_sample = len(terminal_indices_one_less_n) > 0

        truncation_indices_true = np.where(self.is_truncation_stack[indices_upto_next_state])[0]
        first_truncation_index = (
            indices_upto_next_state[truncation_indices_true[0]] if len(truncation_indices_true) > 0 else None
        )

        # the next state can be created if there is no truncation,
        # or if there is a termination (s' not needed) before the first truncation
        is_next_state_valid = (first_truncation_index is None) or (
            first_truncation_index is not None
            and first_terminal_index is not None
            and first_terminal_index <= first_truncation_index
        )

        if not is_state_invalid and is_next_state_valid:
            return self.construct_batch_sample(index, first_terminal_index, is_terminal_for_sample, n, gamma, **kwargs)
        else:
            return None

    def construct_batch_sample(self, index, first_terminal_index, is_terminal_for_sample, n, gamma, **kwargs):
        # if first_terminal_index == None, the sample is a regular sample
        # if first_terminal_index != None, the sample is terminal and the reward should be accumulated until first_terminal_index.
        # Get frames of state of shape (stack_size, H, W) from observation_stack and change to (H, W, stack_size)
        state_index_range = mod_index_range(index - self.stack_size + 1, index, self.max_capacity)
        state = np.moveaxis(self.observation_stack[state_index_range], 0, -1)

        action = self.action_stack[index]
        is_terminal = first_terminal_index is not None

        reward_terms = self.reward_stack[
            mod_index_range(index, first_terminal_index if is_terminal else index + n - 1, self.max_capacity)
        ]
        reward = np.dot(reward_terms, np.power(gamma, np.arange(len(reward_terms))))

        # Get frames of next state of shape (stack_size, H, W) from observation_stack and change to (H, W, stack_size)
        # if is_terminal, then the next state will be ignored so it is irrelevant
        next_state_index_range = mod_index_range(index + n - self.stack_size, index + n - 1, self.max_capacity)
        next_state = np.moveaxis(self.observation_stack[next_state_index_range], 0, -1)

        return ReplayElement(state, action, reward, next_state, is_terminal_for_sample)

    def update(self, indices, loss):
        self.sum_tree.set(indices, np.pow(loss + 1e-10, 0.5))  # Set alpha = 0 for uniform RB
