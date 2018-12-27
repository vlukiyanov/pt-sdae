from ptsdae.model import predict, train
import torch
from torch.utils.data import TensorDataset
from unittest.mock import Mock

# TODO add tests for pretrain, which is admittedly not easy


def test_train_with_prediction():
    autoencoder = Mock()
    autoencoder.return_value = torch.zeros(10, 1000).requires_grad_()
    autoencoder.encode.return_value = torch.zeros(100, 1000)
    optimizer = Mock()
    dataset = TensorDataset(torch.zeros(100, 1000), torch.zeros(100, 1))
    train(
        dataset=dataset,
        autoencoder=autoencoder,
        epochs=1,
        batch_size=10,
        optimizer=optimizer,
        cuda=False
    )
    autoencoder.train.assert_called_once()
    assert autoencoder.call_count == 10
    assert optimizer.zero_grad.call_count == 10
    assert optimizer.step.call_count == 10


def test_train_without_prediction():
    autoencoder = Mock()
    autoencoder.return_value = torch.zeros(10, 1000).requires_grad_()
    autoencoder.encode.return_value = torch.zeros(100, 1000)
    optimizer = Mock()
    dataset = TensorDataset(torch.zeros(100, 1000))
    train(
        dataset=dataset,
        autoencoder=autoencoder,
        epochs=1,
        batch_size=10,
        optimizer=optimizer,
        cuda=False
    )
    autoencoder.train.assert_called_once()
    assert autoencoder.call_count == 10
    assert optimizer.zero_grad.call_count == 10
    assert optimizer.step.call_count == 10


def test_predict_encode():
    # only tests the encode=True
    autoencoder = Mock()
    autoencoder.encode.return_value = torch.zeros(10, 1000)
    dataset = TensorDataset(torch.zeros(100, 1000))
    output = predict(dataset, autoencoder, batch_size=10, cuda=False, encode=True)
    assert autoencoder.encode.call_count == 10
    assert output.shape == (100, 1000)


def test_predict():
    # only tests the encode=False
    autoencoder = Mock()
    autoencoder.return_value = torch.zeros(10, 1000)
    dataset = TensorDataset(torch.zeros(100, 1000))
    output = predict(dataset, autoencoder, batch_size=10, cuda=False, encode=False)
    assert autoencoder.call_count == 10
    assert output.shape == (100, 1000)
