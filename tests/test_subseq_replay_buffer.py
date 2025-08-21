# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/replay_buffer_test.py

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from slimbbf.sample_collection.subseq_replay_buffer import SubsequenceReplayBuffer
from slimbbf.sample_collection import samplers


# Default parameters used when creating the replay memory - mimic Atari.
OBSERVATION_SHAPE = (84, 84)
STACK_SIZE = 4
BATCH_SIZE = 32


def make_state(fill_value: int, stack_size: int):
    return np.full(OBSERVATION_SHAPE + (stack_size,), fill_value)


def make_state_with_values(fill_values: list):
    state = np.zeros(OBSERVATION_SHAPE + (len(fill_values),))
    for i, v in enumerate(fill_values):
        state[:, :, i] = v
    return state


class SubsequenceReplayBufferTest(parameterized.TestCase):

    def testAddUpToCapacity(self):
        capacity = 10
        add_count = 15
        rb = SubsequenceReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=capacity,
            batch_size=BATCH_SIZE,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            spr_window=5,
            stack_size=STACK_SIZE,
            update_horizon=1,
            gamma=1.0,
        )

        for i in range(add_count):
            rb.add(np.full(OBSERVATION_SHAPE, i), i, i, False, False)
        # Since we created the SubsequenceReplayBuffer with a capacity of 10, it should have
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
        rb = SubsequenceReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=10,
            batch_size=BATCH_SIZE,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            spr_window=5,
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

        rb = SubsequenceReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=50,
            batch_size=BATCH_SIZE,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            spr_window=5,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
        )
        for i in range(11):
            rb.add(np.full(OBSERVATION_SHAPE, i), i, i, False, False)

        # ensure that the returned shapes are always correct
        for i in range(3, 7):
            np.testing.assert_array_equal(
                rb._check_valid_and_get_sample(i, 1, 1.0).states_stack.shape, (6,) + OBSERVATION_SHAPE + (4,)
            )
        # ensure that there is the necessary 0 padding
        np.testing.assert_array_equal(zero_state, rb._check_valid_and_get_sample(3, 1, 1.0).states_stack[0, :, :, :3])

        # ensure that after the padding the contents are properly stored
        for i in range(4):
            np.testing.assert_array_equal(
                np.full(OBSERVATION_SHAPE, i), rb._check_valid_and_get_sample(6, 1, 1.0).states_stack[0, :, :, i]
            )

    def testFillBeyondCapacityWithTerminal(self):
        rb = SubsequenceReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=10,
            batch_size=2,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            spr_window=5,
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
        )
        num_adds = 50  # The number of transitions to add to the memory.

        # 40* 41 42 43 44* 45 46 47 48* 49#
        for i in range(num_adds):
            terminal = i % 4 == 0  # Every 4 transitions is terminal.
            rb.add(np.full(OBSERVATION_SHAPE, i), i, i, terminal, False)

        expected_states_stack = lambda start_value: np.array(
            [make_state(40 + (start_value + i) % 10, 1) for i in range(0, 6)]
        )
        expected_actions_stack = lambda start_value: np.array([40 + (start_value + i) % 10 for i in range(0, 6)])
        expected_rewards = np.array([i for i in range(40, 50)])
        expected_terminals = [True if i % 4 == 0 else False for i in range(40, 50)]
        expected_truncations = [False] * 9 + [True]
        expected_next_states = np.array([make_state(i + 1, 1) for i in range(40, 50)])
        expected_same_trajectory_masks = lambda index: (
            np.zeros((6,))
            if index % 4 == 0 or index == 9
            else np.array([1] * ((4 - index) % 4) + [0] * (6 - (4 - index) % 4))
        )

        for i in range(0, 10):
            sample = rb._check_valid_and_get_sample(i, 1, 1.0)
            if expected_truncations[i]:
                self.assertEqual(sample, None)
            else:
                np.testing.assert_array_equal(expected_states_stack(i), sample.states_stack)
                np.testing.assert_array_equal(expected_actions_stack(i), sample.actions_stack)
                np.testing.assert_array_equal(expected_rewards[i], sample.reward)
                np.testing.assert_array_equal(expected_terminals[i], sample.is_terminal)
                np.testing.assert_array_equal(expected_same_trajectory_masks(i), sample.same_trajectory_mask)
                if not expected_terminals[i]:
                    np.testing.assert_array_equal(expected_next_states[i], sample.next_state)

    def testSamplingWithTerminalInTrajectory(self):
        rb = SubsequenceReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=10,
            batch_size=2,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            spr_window=5,
            stack_size=1,
            update_horizon=1,
            gamma=0.5,
        )

        # 0 1 2 3* 4 5 6 7 8 9#
        for i in range(rb._max_capacity):
            rb.add(np.full(OBSERVATION_SHAPE, i), action=i, reward=i, is_terminal=i == 3, is_truncation=False)

        expected_states_stack = lambda start_value: np.array(
            [make_state((start_value + i) % 10, 1) for i in range(0, 6)]
        )
        expected_actions_stack = lambda start_value: np.array([(start_value + i) % 10 for i in range(0, 6)])
        expected_rewards = [3, 6, 5, 3, 15, 18, 21, None, None, None]
        expected_terminals = [True if 1 <= i <= 3 else False for i in range(0, 10)]
        expected_next_states = [
            make_state(3, 1),
            None,
            None,
            None,
            make_state(7, 1),
            make_state(8, 1),
            make_state(9, 1),
            None,
            None,
            None,
        ]
        expected_same_trajectory_masks = lambda index: (
            np.array([1] * (3 - i) + [0] * (3 + i)) if 0 <= index <= 3 else np.array([1] * (9 - i) + [0] * (i - 3))
        )

        for i in range(0, 10):
            sample = rb._check_valid_and_get_sample(i, 3, 1.0)
            if expected_rewards[i] is None:
                self.assertEqual(sample, None)
            else:
                np.testing.assert_array_equal(expected_states_stack(i), sample.states_stack)
                np.testing.assert_array_equal(expected_actions_stack(i), sample.actions_stack)
                np.testing.assert_array_equal(expected_rewards[i], sample.reward)
                np.testing.assert_array_equal(expected_terminals[i], sample.is_terminal)
                np.testing.assert_array_equal(expected_same_trajectory_masks(i), sample.same_trajectory_mask)
                if not expected_terminals[i]:
                    np.testing.assert_array_equal(expected_next_states[i], sample.next_state)

    def testStackSizeWithTerminalAndTruncation(self):
        rb = SubsequenceReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=10,
            batch_size=2,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            spr_window=2,
            stack_size=4,
            update_horizon=1,
            gamma=1,
        )
        # 4 5 6# 0 0 0 7 8 9# 0
        for i in range(rb._max_capacity):
            rb.add(np.full(OBSERVATION_SHAPE, i), action=i, reward=i, is_terminal=i == 3, is_truncation=i == 6)

        for i in range(0, 3):
            np.testing.assert_array_equal(rb._observation_stack[i], make_state(i + 4, 1).squeeze())
        for i in range(3, 6):
            np.testing.assert_array_equal(rb._observation_stack[i], make_state(0, 1).squeeze())
        for i in range(6, 9):
            np.testing.assert_array_equal(rb._observation_stack[i], make_state(i + 1, 1).squeeze())
        np.testing.assert_array_equal(rb._observation_stack[9], make_state(0, 1).squeeze())

        expected_states_stack = (
            [None] * 6
            + [
                np.array(
                    [
                        make_state_with_values([0, 0, 0, 7]),
                        make_state_with_values([0, 0, 7, 8]),
                        make_state_with_values([0, 7, 8, 9]),
                    ]
                ),
                np.array(
                    [
                        make_state_with_values([0, 0, 7, 8]),
                        make_state_with_values([0, 7, 8, 9]),
                        make_state_with_values([7, 8, 9, 0]),
                    ]
                ),
            ]
            + [None] * 2
        )
        expected_actions_stack = [None] * 6 + [np.array([7, 8, 9]), np.array([8, 9, 0])] + [None] * 2
        expected_rewards = [None] * 6 + [7, 8] + [None] * 2
        expected_next_states = (
            [None] * 6 + [make_state_with_values([0, 0, 7, 8]), make_state_with_values([0, 7, 8, 9])] + [None] * 2
        )
        expected_same_trajectory_mask = (
            [None] * 6
            + [
                np.array([1, 1, 0]),
                np.array([1, 0, 0]),
            ]
            + [None] * 2
        )

        for i in range(0, 10):
            sample = rb._check_valid_and_get_sample(i, 1, 1)
            if expected_states_stack[i] is None:
                self.assertEqual(sample, None)
            else:
                np.testing.assert_array_equal(expected_states_stack[i], sample.states_stack)
                np.testing.assert_array_equal(expected_actions_stack[i], sample.actions_stack)
                np.testing.assert_array_equal(expected_rewards[i], sample.reward)
                np.testing.assert_array_equal(False, sample.is_terminal)
                np.testing.assert_array_equal(expected_same_trajectory_mask[i], sample.same_trajectory_mask)
                if expected_next_states[i] is not None:
                    np.testing.assert_array_equal(expected_next_states[i], sample.next_state)

    def testChangingUpdateHorizon(self):
        rb = SubsequenceReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=10,
            batch_size=2,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            spr_window=3,
            stack_size=1,
            update_horizon=1,
            gamma=0.5,
        )

        # 10# 1 2 3 4 5 6# 7 8 9
        for i in range(rb._max_capacity + 1):
            rb.add(np.full(OBSERVATION_SHAPE, i), action=i, reward=i, is_terminal=False, is_truncation=i == 6)

        with self.assertRaises(AssertionError):
            rb.sample(n=6)
        batch, metadata = rb.sample(n=5)
        self.assertNotEqual(None, batch)
        np.testing.assert_equal(batch.states_stack, np.tile([make_state(i, 1) for i in range(1, 5)], (2, 1, 1, 1, 1)))
        np.testing.assert_equal(batch.next_state, np.full((2, *OBSERVATION_SHAPE, 1), 6))
        np.testing.assert_equal(batch.actions_stack, np.tile([1, 2, 3, 4], (2, 1)))
        np.testing.assert_equal(
            batch.reward, np.array([3.5625, 3.5625])
        )  # 1 + 0.5 * 2 + 0.5**2 * 3 + 0.5**3 * 4 + 0.5**4 * 5
        np.testing.assert_equal(batch.is_terminal, np.array([False, False]))
        np.testing.assert_equal(metadata["indices"], np.array([1, 1]))
        np.testing.assert_equal(batch.same_trajectory_mask, np.tile([1, 1, 1, 1], (2, 1)))

    def testChangingGamma(self):
        rb = SubsequenceReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=10,
            batch_size=2,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            spr_window=3,
            stack_size=1,
            update_horizon=1,
            gamma=0.5,
        )

        # 10# 1 2 3 4 5 6# 7 8 9
        for i in range(rb._max_capacity + 1):
            rb.add(np.full(OBSERVATION_SHAPE, i), action=i, reward=i, is_terminal=False, is_truncation=i == 6)

        batch, metadata = rb.sample(n=5, gamma=1)
        self.assertNotEqual(None, batch)
        np.testing.assert_equal(batch.states_stack, np.tile([make_state(i, 1) for i in range(1, 5)], (2, 1, 1, 1, 1)))
        np.testing.assert_equal(batch.next_state, np.full((2, *OBSERVATION_SHAPE, 1), 6))
        np.testing.assert_equal(batch.actions_stack, np.tile([1, 2, 3, 4], (2, 1)))
        np.testing.assert_equal(batch.reward, np.array([15, 15]))
        np.testing.assert_equal(batch.is_terminal, np.array([False, False]))
        np.testing.assert_equal(metadata["indices"], np.array([1, 1]))
        np.testing.assert_equal(batch.same_trajectory_mask, np.tile([1, 1, 1, 1], (2, 1)))

    def testZeroSPRWindow(self):
        rb = SubsequenceReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            max_capacity=10,
            batch_size=2,
            observation_shape=OBSERVATION_SHAPE,
            observation_dtype=np.uint8,
            spr_window=0,
            stack_size=4,
            update_horizon=1,
            gamma=1,
        )
        # 4 5 6# 0 0 0 7 8 9# 0
        for i in range(rb._max_capacity):
            rb.add(np.full(OBSERVATION_SHAPE, i), action=i, reward=i, is_terminal=i == 3, is_truncation=i == 6)

        for i in range(0, 3):
            np.testing.assert_array_equal(rb._observation_stack[i], make_state(i + 4, 1).squeeze())
        for i in range(3, 6):
            np.testing.assert_array_equal(rb._observation_stack[i], make_state(0, 1).squeeze())
        for i in range(6, 9):
            np.testing.assert_array_equal(rb._observation_stack[i], make_state(i + 1, 1).squeeze())
        np.testing.assert_array_equal(rb._observation_stack[9], make_state(0, 1).squeeze())

        expected_states_stack = (
            [None] * 6
            + [np.array([make_state_with_values([0, 0, 0, 7])]), np.array([make_state_with_values([0, 0, 7, 8])])]
            + [None] * 2
        )
        expected_actions_stack = [None] * 6 + [np.array([7]), np.array([8])] + [None] * 2
        expected_rewards = [None] * 6 + [7, 8] + [None] * 2
        expected_next_states = (
            [None] * 6 + [make_state_with_values([0, 0, 7, 8]), make_state_with_values([0, 7, 8, 9])] + [None] * 2
        )
        expected_same_trajectory_mask = [None] * 6 + [np.array([1]), np.array([1])] + [None] * 2

        for i in range(0, 10):
            sample = rb._check_valid_and_get_sample(i, 1, 1)
            if expected_states_stack[i] is None:
                self.assertEqual(sample, None)
            else:
                np.testing.assert_array_equal(expected_states_stack[i], sample.states_stack)
                np.testing.assert_array_equal(expected_actions_stack[i], sample.actions_stack)
                np.testing.assert_array_equal(expected_rewards[i], sample.reward)
                np.testing.assert_array_equal(False, sample.is_terminal)
                np.testing.assert_array_equal(expected_same_trajectory_mask[i], sample.same_trajectory_mask)
                if expected_next_states[i] is not None:
                    np.testing.assert_array_equal(expected_next_states[i], sample.next_state)


if __name__ == "__main__":
    absltest.main()
