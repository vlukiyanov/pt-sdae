import numpy as np
from unittest import TestCase

from ptsdae.utils import cluster_accuracy


class TestClusterAccuracy(TestCase):
    def test_basic(self):
        """
        Basic test to check that the calculation is sensible.
        """
        true_value1 = np.array([1, 2, 1, 2, 0, 0], dtype=np.int64)
        pred_value1 = np.array([2, 1, 2, 1, 0, 0], dtype=np.int64)
        self.assertAlmostEqual(
            cluster_accuracy(true_value1, pred_value1)[1],
            1.0
        )
        self.assertAlmostEqual(
            cluster_accuracy(true_value1, pred_value1, 3)[1],
            1.0
        )
        true_value2 = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
        pred_value2 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        self.assertAlmostEqual(
            cluster_accuracy(true_value2, pred_value2)[1],
            1.0 / 6.0
        )
        self.assertAlmostEqual(
            cluster_accuracy(true_value2, pred_value2, 6)[1],
            1.0 / 6.0
        )
