# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/replay_buffer_test.py

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from slimdqn.sample_collection import replay_buffer
from slimdqn.sample_collection.replay_buffer import ReplayElement
from slimdqn.sample_collection import samplers


# Default parameters used when creating the replay memory - mimic Atari.
OBSERVATION_SHAPE = (84, 84)
STACK_SIZE = 4
BATCH_SIZE = 32


class ReplayBufferTest(parameterized.TestCase):

    def testAddUpToCapacity(self):
        capacity = 10
        add_count = 15
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=capacity,
            batch_size=BATCH_SIZE,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            stack_size=STACK_SIZE,
            update_horizon=1,
            gamma=1.0,
        )

        for i in range(add_count):
            rb.add(np.full(OBSERVATION_SHAPE, i), i, i, False, False)
        # Since we created the ReplayBuffer with a capacity of 10, it should have
        # gotten rid of the first 6 elements added.
        expected_keys = [7, 8, 9, 10, 11, 12, 13, 14, 5, 6]
        for idx, key in enumerate(expected_keys):
            np.testing.assert_array_equal(rb._observation_stack[idx], np.full(OBSERVATION_SHAPE, key))
            self.assertEqual(rb._action_stack[idx], key)
            self.assertEqual(rb._reward_stack[idx], key)
            self.assertEqual(rb._is_terminal_stack[idx], False)
            if idx == capacity - 3:
                self.assertEqual(rb._is_truncation_stack[idx], True)
            else:
                self.assertEqual(rb._is_truncation_stack[idx], False)

    def testNSteprewards(self):
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=10,
            batch_size=BATCH_SIZE,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
        )

        for i in range(50):
            # add non-terminating observations with reward 2
            rb.add(np.full(OBSERVATION_SHAPE, i), 0, 2.0, False, False)

        for _ in range(100):
            batch, _ = rb.sample()
            # Make sure the total reward is reward per step x update_horizon.
            np.testing.assert_array_equal(batch.reward, np.ones(BATCH_SIZE) * 10.0)

    def testGetStack(self):
        zero_state = np.zeros(OBSERVATION_SHAPE + (3,))

        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=50,
            batch_size=BATCH_SIZE,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
        )
        for i in range(11):
            rb.add(np.full(OBSERVATION_SHAPE, i), i, i, False, False)

        # ensure that the returned shapes are always correct
        for i in range(3, 7):
            np.testing.assert_array_equal(rb._check_valid_and_get_sample(i, 1, 1.0)[0].shape, OBSERVATION_SHAPE + (4,))

        # ensure that there is the necessary 0 padding
        state = rb._check_valid_and_get_sample(3, 1, 1.0)[0]
        np.testing.assert_array_equal(zero_state, state[:, :, :3])

        # ensure that after the padding the contents are properly stored
        state = rb._check_valid_and_get_sample(6, 1, 1.0)[0]
        for i in range(4):
            np.testing.assert_array_equal(np.full(OBSERVATION_SHAPE, i), state[:, :, i])

    def testSampleTransitionBatch(self):
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=10,
            batch_size=2,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
        )
        num_adds = 50  # The number of transitions to add to the memory.

        for i in range(num_adds):
            terminal = i % 4 == 0  # Every 4 transitions is terminal.
            rb.add(np.full(OBSERVATION_SHAPE, i), i, i, terminal, False)

        def make_state(key: int):
            return np.full(OBSERVATION_SHAPE + (1,), key)

        expected_states = np.array([make_state(i) for i in range(40, 50)])
        expected_actions_and_rewards = np.arange(40, 50)
        expected_terminals = [True if i % 4 == 0 else False for i in range(40, 50)]
        expected_truncations = [False] * 9 + [True]
        expected_next_states = np.array([make_state(i + 1) for i in range(40, 50)])

        for i in range(0, 10):
            sample = rb._check_valid_and_get_sample(i, 1, 0.99)
            if expected_truncations[i]:
                self.assertEqual(sample, None)
            else:
                np.testing.assert_array_equal(expected_states[i], sample[0])
                np.testing.assert_array_equal(expected_actions_and_rewards[i], sample[1])
                np.testing.assert_array_equal(expected_actions_and_rewards[i], sample[2])
                np.testing.assert_array_equal(expected_terminals[i], sample[4])
                if not expected_terminals[i]:
                    np.testing.assert_array_equal(expected_next_states[i], sample[3])

