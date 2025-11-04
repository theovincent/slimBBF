# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/replay_buffer_test.py

import unittest
import numpy as np

from slimbbf.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer


def make_state(fill_values: list):
    # creates a state of shape (84, 84, len(fill_values)) with values given by fill_values
    state = np.zeros((84, 84) + (len(fill_values),))
    for i, v in enumerate(fill_values):
        state[:, :, i] = v
    return state


def make_states_stack(fill_values: list, stack_size: int):
    # creates a states stack of shape (K, 84, 84, stack_size) with K determined by amount of fill values and stack_size
    states_stack = []
    for i in range(stack_size, len(fill_values) + 1):
        states_stack.append(make_state(fill_values[i - stack_size : i]))
    return np.array(states_stack)


class SubsequenceReplayBufferTest(unittest.TestCase):

    def testGetStack(self):
        rb = SubsequenceReplayBuffer(
            max_capacity=10,
            seed=0,
            batch_size=32,
            observation_shape=(84, 84),
            observation_dtype=np.uint8,
            spr_window=3,
            stack_size=4,
            clipping=None,
        )
        rb.add(np.full((84, 84), 1), 1, 1, False, False)
        rb.add(np.full((84, 84), 2), 2, 2, False, False)
        rb.add(np.full((84, 84), 3), 3, 3, False, False)
        # After adding RB is [0 0 0 1 2 3#] (# = truncation)

        # ensure that the returned state at index 3 and 4 is correct
        np.testing.assert_array_equal(
            make_states_stack([0, 0, 0, 1, 2, 3, 0], 4), rb.check_valid_and_get_sample(3, 1, 1).states_stack
        )
        np.testing.assert_array_equal(
            make_states_stack([0, 0, 1, 2, 3, 0, 0], 4), rb.check_valid_and_get_sample(4, 1, 1).states_stack
        )

        # ensure that the returned action, reward, next_state, is_terminal are correct
        sample = rb.check_valid_and_get_sample(3, 1, 1)
        np.testing.assert_array_equal([1, 2, 3, 0], sample.actions_stack)
        np.testing.assert_array_equal(make_state([0, 0, 1, 2]), sample.next_state)
        self.assertEqual(sample.reward, 1)
        self.assertEqual(sample.is_terminal, False)
        np.testing.assert_array_equal([1, 1, 1, 0], sample.same_trajectory_mask)

        # ensure indices 2,5 are invalid
        self.assertEqual(rb.check_valid_and_get_sample(2, 1, 1), None)
        self.assertEqual(rb.check_valid_and_get_sample(5, 1, 1), None)

    def testFillBeyondCapacityWithTerminal(self):
        rb = SubsequenceReplayBuffer(
            max_capacity=10,
            seed=0,
            batch_size=2,
            observation_shape=(84, 84),
            observation_dtype=np.uint8,
            spr_window=5,
            stack_size=1,  # testing with stack size 1
            clipping=None,
        )

        # After adding 50 transitions, RB would contain transitions [40* 41 42 43 44* 45 46 47 48* 49#] (* = terminal, # = truncation)
        for i in range(50):
            terminal = i % 4 == 0  # Every 4 transitions is terminal.
            rb.add(np.full((84, 84), i), i, i, terminal, False)

        # index 9 is invalid as we cannot create next_state
        self.assertEqual(rb.check_valid_and_get_sample(9, 1, 1.0), None)

        # check valid non-terminating sample at index 2 (s=[42,...,47], a=[42,...,47], r=42 + 0.5*43, s'=44, d=False, mask=[1,1,1,0,0,0]) (for n=2)
        sample = rb.check_valid_and_get_sample(2, 2, 0.5)  # n=2, gamma=0.5
        np.testing.assert_array_equal(make_states_stack([42, 43, 44, 45, 46, 47], 1), sample.states_stack)
        np.testing.assert_array_equal([42, 43, 44, 45, 46, 47], sample.actions_stack)
        self.assertEqual(42 + 0.5 * 43, sample.reward)
        self.assertEqual(False, sample.is_terminal)
        np.testing.assert_array_equal(make_state([44]), sample.next_state)
        np.testing.assert_array_equal([1, 1, 1, 0, 0, 0], sample.same_trajectory_mask)

        # check valid terminating sample at index 8 (s=[48,49,40,...,43], a=[48,49,40,...,43], r=48, d=True, mask=[1,0,0..]) (s' doesn't matter here)
        sample = rb.check_valid_and_get_sample(8, 1, 1.0)  # gamma=1.0
        np.testing.assert_array_equal(make_states_stack([48, 49, 40, 41, 42, 43], 1), sample.states_stack)
        np.testing.assert_array_equal([48, 49, 40, 41, 42, 43], sample.actions_stack)
        self.assertEqual(48, sample.reward)
        self.assertEqual(True, sample.is_terminal)
        np.testing.assert_array_equal([1, 0, 0, 0, 0, 0], sample.same_trajectory_mask)
