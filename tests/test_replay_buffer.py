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
        rb = ReplayBuffer(
            max_capacity=5,
            seed=0,
            batch_size=32,
            observation_shape=(84, 84),
            observation_dtype=np.uint8,
            stack_size=4,
            clipping=None,
        )

        for i in range(5):
            rb.add(np.full((84, 84), i), i, i, False, False)
        # RB starts with [0 0 0# - -] as keys (# = truncation)
        # After adding obs 0,1,2,3,4 it becomes [2 3 4# 0 1]
        expected_keys = [2, 3, 4, 0, 1]
        for idx, key in enumerate(expected_keys):
            np.testing.assert_array_equal(rb.observation_stack[idx], np.full((84, 84), key))
            self.assertEqual(rb.action_stack[idx], key)
            self.assertEqual(rb.reward_stack[idx], key)
            self.assertEqual(rb.is_terminal_stack[idx], False)
            if idx == 2:
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

    def testUpdateAndSample(self):
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
