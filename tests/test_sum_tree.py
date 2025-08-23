# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/sum_tree_test.py

import unittest
import numpy as np

from slimbbf.sample_collection import sum_tree


class SumTreeTest(unittest.TestCase):

    def setUp(self):
        super(SumTreeTest, self).setUp()
        self.tree = sum_tree.SumTree(capacity=100)

    def test_set_small_capacity(self):
        tree = sum_tree.SumTree(capacity=1)
        tree.set(0, 1.5)
        self.assertEqual(tree.root, 1.5)

    def test_set_and_get_value(self):
        self.tree.set(0, 1.0)
        self.assertEqual(self.tree.get(0), 1.0)

        # Validate that all nodes on the leftmost branch have value 1.
        leaf_index = self.tree.first_leaf_offset
        while leaf_index > 0:
            leaf_index = leaf_index // 2
            self.assertEqual(self.tree.nodes[leaf_index], 1.0)

    def test_set_and_get_values_vectorized(self):
        self.tree.set(
            np.array([1, 2], dtype=np.int32),
            np.array([3.0, 4.0], dtype=np.float32),
        )
        self.assertEqual(self.tree.get(1), 3.0)
        self.assertEqual(self.tree.get(2), 4.0)
        self.assertEqual(self.tree.root, 7.0)

    def test_set_with_duplicates(self):
        self.tree.set(
            np.array([1, 1, 1, 2, 2], dtype=np.int32),
            np.array([3.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float32),
        )
        self.assertEqual(self.tree.get(1), 3.0)
        self.assertEqual(self.tree.get(2), 4.0)
        self.assertEqual(self.tree.root, 7.0)

    def test_capacity_greater_than_requested(self):
        self.assertGreaterEqual(self.tree.nodes.size, 100)

    def test_query_value(self):
        self.tree.set(5, 1.0)
        self.assertEqual(self.tree.query(0.99), 5)

    def test_query_values_vectorized(self):
        """
              [2.5]
           [1.5]  [1.0]
        [0.5 1.0 0.5 0.5]
        """
        tree = sum_tree.SumTree(capacity=4)
        tree.set(
            np.array([0, 1, 2, 3], dtype=np.int32),
            np.array([0.5, 1.0, 0.5, 0.5], dtype=np.float32),
        )
        self.assertEqual(tree.root, 2.5)
        self.assertEqual(tree.depth, 3)
        self.assertEqual(tree.nodes.size, 7)
        np.testing.assert_array_equal(
            tree.query(np.array([1.5, 1.0])),
            np.array([2, 1], np.int32),
        )

    def test_update_sum_values(self):
        """
              [2.5]
           [1.5]  [1.0]
        [0.5 1.0 0.5 0.5]
        """
        tree = sum_tree.SumTree(capacity=4)
        tree.set(
            np.array([0, 1, 2, 3], dtype=np.int32),
            np.array([0.5, 1.0, 0.5, 0.5], dtype=np.float32),
        )
        tree.set(0, 0.25)
        self.assertEqual(tree.root, 2.25)
        self.assertEqual(tree.query(0.249), 0)
        self.assertEqual(tree.query(0.5), 1)
        self.assertEqual(tree.query(1.25), 2)

    def test_query_values_vectorized_large_tree(self):
        """
                   [8]
             [4]         [4]
          [2]   [2]   [2]   [2]
        [1, 1, 1, 1, 1, 1, 1, 1]
        """
        tree = sum_tree.SumTree(capacity=8)
        tree.set(
            np.arange(8, dtype=np.int32),
            np.ones((8,), dtype=np.float32),
        )
        self.assertEqual(tree.root, 8.0)
        self.assertEqual(tree.depth, 4)
        self.assertEqual(tree.nodes.size, 15)
        np.testing.assert_array_equal(
            tree.query(np.arange(8, dtype=np.int32)),
            np.arange(8, dtype=np.int32),
        )

    def test_max_recorded_priority(self):
        k = 32
        self.tree.set(0, 0)
        self.assertEqual(self.tree.max_recorded_priority, 1)
        for i in range(1, k):
            self.tree.set(i, i)
            self.assertEqual(self.tree.max_recorded_priority, i)
