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
            batch_size=BATCH_SIZE,
            observation_shape=OBSERVATION_SHAPE,
            max_capacity=capacity,
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
            batch_size=BATCH_SIZE,
            max_capacity=10,
            observation_shape=OBSERVATION_SHAPE,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
        )

        for i in range(50):
            # add non-terminating observations with reward 2
            rb.add(np.full(OBSERVATION_SHAPE, i), 0, 2.0, False, False)

        for _ in range(100):
            batch = rb.sample()
            # Make sure the total reward is reward per step x update_horizon.
            np.testing.assert_array_equal(batch.reward, np.ones(BATCH_SIZE) * 10.0)
