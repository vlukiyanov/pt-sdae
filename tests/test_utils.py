import numpy as np
import torch
from unittest import TestCase

from ptsdae.utils import cluster_accuracy, pretrain_accuracy, Classifier


class TestClusterAccuracy(TestCase):
    def test_basic(self):
        """
        Basic test to check that the calculation is sensible.
        """
        true_value1 = np.array([1, 2, 1, 2, 0, 0], dtype=np.int64)
        pred_value1 = np.array([2, 1, 2, 1, 0, 0], dtype=np.int64)
        self.assertAlmostEqual(cluster_accuracy(true_value1, pred_value1)[1], 1.0)
        self.assertAlmostEqual(cluster_accuracy(true_value1, pred_value1, 3)[1], 1.0)
        self.assertDictEqual(
            cluster_accuracy(true_value1, pred_value1)[0], {0: 0, 1: 2, 2: 1}
        )
        true_value2 = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
        pred_value2 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        self.assertAlmostEqual(cluster_accuracy(true_value2, pred_value2)[1], 1.0 / 6.0)
        self.assertAlmostEqual(
            cluster_accuracy(true_value2, pred_value2, 6)[1], 1.0 / 6.0
        )
        true_value3 = np.array([1, 3, 1, 3, 0, 2], dtype=np.int64)
        pred_value3 = np.array([2, 1, 2, 1, 3, 0], dtype=np.int64)
        self.assertDictEqual(
            cluster_accuracy(true_value3, pred_value3)[0], {2: 1, 1: 3, 3: 0, 0: 2}
        )


class TestPretrainAccuracy(TestCase):
    def test_basic(self):
        output_tensor = torch.ones(10)
        input_tensor = torch.ones(10)
        self.assertAlmostEqual(pretrain_accuracy(output_tensor, input_tensor), 1.0)
        self.assertAlmostEqual(pretrain_accuracy(input_tensor, output_tensor), 1.0)

    def test_mixed(self):
        output_tensor = torch.ones(10)
        output_tensor[5:] = 0
        input_tensor = torch.ones(10)
        self.assertAlmostEqual(pretrain_accuracy(output_tensor, input_tensor), 0.5)
        self.assertAlmostEqual(pretrain_accuracy(input_tensor, output_tensor), 0.5)


class TestClassifier(TestCase):
    def test_basic(self):
        classifier = Classifier([10, 100, 1000, 3])
        input_tensor = torch.ones(10, 10)
        self.assertEqual(classifier(input_tensor).shape, (10, 3))
