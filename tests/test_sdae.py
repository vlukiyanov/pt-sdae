import torch
from unittest import TestCase

from ptsdae.sdae import build_units, StackedDenoisingAutoEncoder


class TestBuildUnits(TestCase):
    def test_basic(self):
        units1 = build_units([100, 10], torch.nn.ReLU())
        units3 = build_units([100, 10, 5, 2], torch.nn.ReLU())
        units5 = build_units([100, 10, 5, 2, 2, 2], torch.nn.ReLU())
        self.assertEqual(len(units1), 1)
        self.assertEqual(len(units3), 3)
        self.assertEqual(len(units5), 5)
        for units in [units1, units3, units5]:
            for item in units:
                self.assertTrue(hasattr(item, 'linear'))
                self.assertTrue(hasattr(item, 'activation'))
                self.assertIsInstance(item.linear, torch.nn.Linear)
                self.assertIsInstance(item.activation, torch.nn.ReLU)

    def test_arguments(self):
        units = build_units(
            [100, 10, 5, 2],
            None,
        )
        for item in units:
            self.assertTrue(hasattr(item, 'linear'))
            self.assertFalse(hasattr(item, 'activation'))
            self.assertIsInstance(item.linear, torch.nn.Linear)


class TestStackedDenoisingAutoEncoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dimensions = list(reversed(range(5, 11)))
        cls.ae = StackedDenoisingAutoEncoder(cls.dimensions)

    def test_basic(self):
        self.assertEqual(self.ae.decoder(torch.Tensor(1, 5).fill_(0.0)).shape, (1, 10))
        self.assertEqual(self.ae.encoder(torch.Tensor(1, 10).fill_(0.0)).shape, (1, 5))
        self.assertEqual(self.ae.forward(torch.Tensor(1, 10).fill_(0.0)).shape, (1, 10))

    def test_get_stack(self):
        for index in range(0, len(self.dimensions) - 1):
            encoder, decoder = self.ae.get_stack(index)
            self.assertIsInstance(encoder, torch.nn.Linear)
            self.assertIsInstance(decoder, torch.nn.Linear)
        with self.assertRaises(ValueError):
            self.ae.get_stack(len(self.dimensions))
        with self.assertRaises(ValueError):
            self.ae.get_stack(-1)
