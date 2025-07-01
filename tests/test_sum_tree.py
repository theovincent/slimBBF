# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/sum_tree_test.py
"""Tests for vectorized sum tree."""

from absl.testing import absltest
from absl.testing import parameterized
from slimdqn.sample_collection import sum_tree
import numpy as np
import jax


class SumTreeTest(parameterized.TestCase):

    def setUp(self):
        super(SumTreeTest, self).setUp()
        self._tree = sum_tree.SumTree(capacity=100)

    def test_negative_capacity_raises(self):
        with self.assertRaises(AssertionError):
            sum_tree.SumTree(capacity=-1)

    def test_negative_value_raises(self):
        with self.assertRaises(AssertionError):
            self._tree.set(0, -1)

    def test_set_small_capacity(self):
        tree = sum_tree.SumTree(capacity=1)
        tree.set(0, 1.5)
        self.assertEqual(tree.nodes[0], 1.5)

    def test_set_and_get_value(self):
        self._tree.set(0, 1.0)
        self.assertEqual(self._tree.get(0), 1.0)

        # Validate that all nodes on the leftmost branch have value 1.
        leaf_index = self._tree._first_leaf_offset
        while leaf_index > 0:
            leaf_index = leaf_index // 2
            self.assertEqual(self._tree.nodes[leaf_index], 1.0)

    def test_set_and_get_values(self):
        self._tree.set(1, 3.0)
        self._tree.set(2, 4.0)
        self.assertEqual(self._tree.get(1), 3.0)
        self.assertEqual(self._tree.get(2), 4.0)
        self.assertEqual(self._tree.nodes[0], 7.0)

    def test_set_with_duplicates(self):
        for index, priority in zip([1, 1, 1, 2, 2], [3.0, 3.0, 3.0, 4.0, 4.0]):
            self._tree.set(index, priority)
        self.assertEqual(self._tree.get(1), 3.0)
        self.assertEqual(self._tree.get(2), 4.0)
        self.assertEqual(self._tree.nodes[0], 7.0)

    def test_capacity_greater_than_requested(self):
        self.assertGreaterEqual(self._tree.nodes.size, 100)

    def test_sample_empty_tree(self):
        with self.assertRaises(AssertionError):
            self._tree.sample(1, jax.random.PRNGKey(0))

    def test_query_value(self):
        self._tree.set(5, 1.0)
        self.assertEqual(self._tree.query(0.99), 5)

    def test_query_values_vectorized(self):
        #
        """
              [2.5]
           [1.5]  [1.0]
        [0.5 1.0 0.5 0.5]
        """
        tree = sum_tree.SumTree(capacity=4)
        for index, priority in zip(
            np.array([0, 1, 2, 3], dtype=np.int32), np.array([0.5, 1.0, 0.5, 0.5], dtype=np.float32)
        ):
            tree.set(index, priority)
        self.assertEqual(tree.nodes[0], 2.5)
        self.assertEqual(tree.depth, 2)
        self.assertEqual(tree.nodes.size, 7)
        np.testing.assert_array_equal(
            [tree.query(i) for i in np.array([0.6, 0.4])],
            np.array([2, 1], np.int32),
        )

    def test_update_sum_values(self):
        #
        """.

              [2.5]
           [1.5]  [1.0]
        [0.5 1.0 0.5 0.5]
        """
        tree = sum_tree.SumTree(capacity=4)
        for index, priority in zip(
            np.array([0, 1, 2, 3], dtype=np.int32), np.array([0.5, 1.0, 0.5, 0.5], dtype=np.float32)
        ):
            tree.set(index, priority)
        tree.set(0, 0.25)
        self.assertEqual(tree.nodes[0], 2.25)
        self.assertEqual(tree.query(0.111), 0)
        self.assertEqual(tree.query(0.222), 1)
        self.assertEqual(tree.query(0.556), 2)

    def test_query_values_vectorized_large_tree(self):
        #
        """
                   [8]
             [4]         [4]
          [2]   [2]   [2]   [2]
        [1, 1, 1, 1, 1, 1, 1, 1]
        """
        tree = sum_tree.SumTree(capacity=8)
        for index, priority in zip(np.arange(8, dtype=np.int32), np.ones((8,), dtype=np.float32)):
            tree.set(index, priority)
        self.assertEqual(tree.nodes[0], 8.0)
        self.assertEqual(tree.depth, 3)
        self.assertEqual(tree.nodes.size, 15)
        np.testing.assert_array_equal(
            [tree.query(i) for i in np.linspace(start=0, stop=1, num=8)],
            np.arange(8, dtype=np.int32),
        )

    def test_max_recorded_priority(self):
        k = 32
        self._tree.set(0, 0)
        self.assertEqual(self._tree.max_recorded_priority, 1)
        for i in range(1, k):
            self._tree.set(i, i)
            self.assertEqual(self._tree.max_recorded_priority, i)


if __name__ == "__main__":
    absltest.main()
