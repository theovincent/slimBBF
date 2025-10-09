# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/replay_buffer_test.py

import unittest
import numpy as np

from slimbbf.sample_collection.replay_buffer import ReplayBuffer


def make_state(fill_values: list):
    # creates a state of shape (84, 84, len(fill_values)) with values given by fill_values
    state = np.zeros((84, 84) + (len(fill_values),))
    for i, v in enumerate(fill_values):
        state[:, :, i] = v
    return state


class ReplayBufferTest(unittest.TestCase):

    def testAddUpToCapacity(self):
        capacity = 5
        add_count = 5
        rb = ReplayBuffer(
            max_capacity=capacity,
            seed=0,
            batch_size=32,
            observation_shape=(84, 84),
            observation_dtype=np.uint8,
            stack_size=4,
            clipping=None,
        )

        for i in range(add_count):
            rb.add(np.full((84, 84), i), i, i, False, False)
        # RB starts with [0 0 0# - -] as keys (# = truncation)
        # After adding obs 0,1,2,3,4 it becomes [2 3 4 0 1]
        expected_keys = [2, 3, 4, 0, 1]
        for idx, key in enumerate(expected_keys):
            np.testing.assert_array_equal(rb.observation_stack[idx], np.full((84, 84), key))
            self.assertEqual(rb.action_stack[idx], key)
            self.assertEqual(rb.reward_stack[idx], key)
            self.assertEqual(rb.is_terminal_stack[idx], False)
            if idx == capacity - 3:
                self.assertEqual(rb.is_truncation_stack[idx], True)
            else:
                self.assertEqual(rb.is_truncation_stack[idx], False)

    def testNSteprewards(self):
        rb = ReplayBuffer(
            max_capacity=10,
            seed=0,
            batch_size=32,
            observation_shape=(84, 84),
            observation_dtype=np.uint8,
            stack_size=4,
            clipping=None,
        )

        for i in range(50):
            # add non-terminating observations with reward 2
            rb.add(np.full((84, 84), i), 0, 2.0, False, False)

        for _ in range(100):
            batch, _, _ = rb.sample(n=5, gamma=1, n_batches=1)
            # Make sure the total reward is reward per step x update_horizon.
            np.testing.assert_array_equal(batch[0].reward, np.ones(32) * 10.0)

    def testGetStack(self):
        rb = ReplayBuffer(
            max_capacity=10,
            seed=0,
            batch_size=32,
            observation_shape=(84, 84),
            observation_dtype=np.uint8,
            stack_size=4,
            clipping=None,
        )
        rb.add(np.full((84, 84), 1), 1, 1, False, False)
        rb.add(np.full((84, 84), 2), 2, 2, False, False)
        rb.add(np.full((84, 84), 3), 3, 3, False, False)
        # After adding RB is [0 0 0 1 2 3#]

        # ensure that the returned state at index 3 and 4 is correct
        np.testing.assert_array_equal(rb.check_valid_and_get_sample(3, 1, 1).state, make_state([0, 0, 0, 1]))
        np.testing.assert_array_equal(rb.check_valid_and_get_sample(4, 1, 1).state, make_state([0, 0, 1, 2]))

        # ensure that the returned action, reward, next_state, is_terminal are correct
        sample = rb.check_valid_and_get_sample(3, 1, 1)
        self.assertEqual(sample.action, 1)
        np.testing.assert_array_equal(sample.next_state, make_state([0, 0, 1, 2]))
        self.assertEqual(sample.reward, 1)
        self.assertEqual(sample.is_terminal, False)

        # ensure indices 2,5 are invalid
        self.assertEqual(rb.check_valid_and_get_sample(2, 1, 1), None)
        self.assertEqual(rb.check_valid_and_get_sample(5, 1, 1), None)

    def testFillBeyondCapacityWithTerminal(self):
        rb = ReplayBuffer(
            max_capacity=10,
            seed=0,
            batch_size=2,
            observation_shape=(84, 84),
            observation_dtype=np.uint8,
            stack_size=1,  # testing with stack size 1
            clipping=None,
        )
        num_adds = 50  # The number of transitions to add to the memory.

        # After adding 50 transitions, RB would contain transitions [40* 41 42 43 44* 45 46 47 48* 49#] (* = terminal, # = truncation)
        for i in range(num_adds):
            terminal = i % 4 == 0  # Every 4 transitions is terminal.
            rb.add(np.full((84, 84), i), i, i, terminal, False)

        # index 9 is invalid as we cannot create next_state
        self.assertEqual(rb.check_valid_and_get_sample(9, 1, 1.0), None)

        # check valid sample at index 2 (s=42, a=42, r=42 + 0.5*43, s'=44, d=False) (for n=2)
        sample = rb.check_valid_and_get_sample(2, 2, 0.5)  # n=2, gamma=0.5
        np.testing.assert_array_equal(make_state([42]), sample.state)
        self.assertEqual(42, sample.action)
        self.assertEqual(42 + 0.5 * 43, sample.reward)
        self.assertEqual(False, sample.is_terminal)
        np.testing.assert_array_equal(make_state([44]), sample.next_state)

        # check valid sample at index 8 (s=48, a=48, r=48, d=True) (s' doesn't matter here)
        sample = rb.check_valid_and_get_sample(8, 1, 1.0)  # gamma=1.0
        np.testing.assert_array_equal(make_state([48]), sample.state)
        self.assertEqual(48, sample.action)
        self.assertEqual(48, sample.reward)
        self.assertEqual(True, sample.is_terminal)

    def testUpdateAndSmaple(self):
        rb = ReplayBuffer(
            max_capacity=10,
            seed=0,
            batch_size=32,
            observation_shape=(84, 84),
            observation_dtype=np.uint8,
            stack_size=1,  # testing with stack size 1
            clipping=None,
        )

        for i in range(5):
            rb.add(np.full((84, 84), i), i, i, False, False)

        self.assertEqual(rb.sum_tree.root, 5)
        rb.update(np.array([0, 1, 2, 3, 4]), np.array([1.0, 4.0, 9.0, 16.0, 0.0]))  # updates priorities to 1,2,3,4,0
        self.assertAlmostEqual(rb.sum_tree.root, 10, places=4)

        # test if zero priority absent
        _, indices, _ = rb.sample(1, 0.5, n_batches=1)
        np.testing.assert_array_less(indices, 4)  # 0 priority should not be sampled

        indices = np.array([5, 6, 7, 8, 9, 0, 1])
        for i in range(5, 15):
            rb.add(np.full((84, 84), i), i, i, False, False)  # now priorities should be 4,...,4 (max so far is 4)
        self.assertAlmostEqual(rb.sum_tree.root, 40, places=4)
        _, _, importance_weights = rb.sample(1, 0.5, n_batches=1)
        np.testing.assert_array_equal(importance_weights, np.ones_like(importance_weights))
