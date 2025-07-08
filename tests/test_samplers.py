# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/samplers_test.py
"""Testing samplers."""

from absl.testing import absltest
from absl.testing import parameterized
from slimdqn.sample_collection import samplers
import numpy as np


class UniformSamplingTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.sampler = samplers.UniformSamplingDistribution(seed=0)

    def test_sample(self):
        indices = np.array([0, 1, 2, 3, 4])

        with self.assertRaises(ValueError):
            self.sampler.sample(5)
        for index in indices:
            self.sampler.add(index)

        samples = self.sampler.sample(20)
        self.assertEqual(self.sampler.index_max, 4)
        self.assertEqual(samples.min(), 0)
        self.assertEqual(samples.max(), 4)

        indices = np.array([5, 6, 7, 8, 9, 0])
        for index in indices:
            self.sampler.add(index)
        samples = self.sampler.sample(50)
        self.assertEqual(self.sampler.index_max, 9)
        self.assertEqual(samples.min(), 0)
        self.assertEqual(samples.max(), 9)


class PrioritizedSamplingTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.sampler = samplers.PrioritizedSamplingDistribution(seed=0, max_capacity=10)

    def test_sample(self):
        indices = np.array([0, 1, 2, 3, 4])
        losses = np.array([1.0, 4.0, 9.0, 16.0, 0.0])

        for index in indices:
            self.sampler.add(index)

        self.assertEqual(self.sampler._sum_tree.root, 5)
        self.sampler.update({"indices": indices, "loss": losses})
        self.assertEqual(self.sampler._sum_tree.root, 10)

        # test if zero priority absent
        samples = self.sampler.sample(5)
        np.testing.assert_array_less(samples, 4)
        np.testing.assert_array_equal(self.sampler.get_probabilities(samples), (samples + 1) / 10)

        self.sampler.update({"indices": np.array([2, 3]), "loss": np.array([0.0, 0.0])})

        # test if priority updated properly
        samples = self.sampler.sample(5)
        np.testing.assert_array_less(samples, 2)
        np.testing.assert_array_equal(self.sampler.get_probabilities(samples), (samples + 1) / 3)

        indices = np.array([5, 6, 7, 8, 9, 0, 1])
        for index in indices:
            self.sampler.add(index)
        self.assertEqual(self.sampler._sum_tree.root, 28)
        samples = self.sampler.sample(20)
        self.assertEqual(np.bitwise_and(samples >= 2, samples <= 4).any(), 0)
        np.testing.assert_array_equal(self.sampler.get_probabilities(samples), np.ones_like(samples) / 7)


if __name__ == "__main__":
    absltest.main()
