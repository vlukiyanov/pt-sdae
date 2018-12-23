# pt-sdae
[![Build Status](https://travis-ci.org/vlukiyanov/pt-sdae.svg?branch=master)](https://travis-ci.org/vlukiyanov/pt-sdae) [![codecov](https://codecov.io/gh/vlukiyanov/pt-sdae/branch/master/graph/badge.svg)](https://codecov.io/gh/vlukiyanov/pt-sdae)

PyTorch implementation of a version of the Stacked Denoising AutoEncoder. Compatible with PyTorch 1.0.0 and Python 3.6 or 3.7 with or without CUDA.

## Examples

An example using MNIST data can be found in the `examples/mnist/mnist.py` which achieves around 80% accuracy using
k-Means on the encoded values.

Here is an example [confusion matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html), true labels on y-axis and predicted labels on the x-axis.
![Alt text](confusion.png)

## Usage

This is distributed as a Python package `ptsdae` and can be installed with `python setup.py install`. The PyTorch `nn.Module` class representing the SDAE is `StackedDenoisingAutoEncoder` in `ptsdae.sdae`, while the `pretrain` and `train` functions from `ptsdae.model` are used to train the autoencoder.

Currently this code is used in a PyTorch implementation of DEC, see https://github.com/vlukiyanov/pt-dec.
