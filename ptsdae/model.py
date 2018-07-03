from typing import Any, Callable, Optional, Union
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ptsdae.dae import DenoisingAutoencoder
from ptsdae.sdae import StackedDenoisingAutoEncoder
from ptsdae.utils import SimpleDataset, pretrain_accuracy


def train(dataset: torch.utils.data.Dataset,
          autoencoder: torch.nn.Module,
          epochs: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
          scheduler: Any = None,
          validation: Optional[torch.utils.data.Dataset] = None,
          corruption: Optional[float] = None,
          cuda: bool = True,
          sampler: Optional[torch.utils.data.sampler.Sampler] = None,
          silent: bool = False,
          update_freq: Optional[int] = None,
          update_callback: Optional[Callable[[float, float], None]] = None,
          epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None) -> None:
    """
    Function to train an autoencoder using the provided dataset. If the dataset consists of 2-tuples or lists of
    (feature, prediction), then the prediction is stripped away.

    :param dataset: training Dataset
    :param autoencoder: autoencoder to train
    :param epochs: number of training epochs
    :param batch_size: batch size for training
    :param optimizer: optimizer to use
    :param scheduler: scheduler to use, or None to disable, defaults to None
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param validation: instance of Dataset to use for validation, set to None to disable, defaults to None
    :param cuda: whether CUDA is used, defaults to True
    :param sampler: sampler to use in the DataLoader, set to None to disable, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, set to None disables, default 1/10 of epochs
    :param update_callback: optional function of accuracy and loss to update
    :param epoch_callback: optional function of epoch and model
    :return: None
    """
    if update_freq is None:
        update_freq = max(epochs // 10, 1)  # default 1/10 of epochs
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        sampler=sampler,
        shuffle=True
    )
    if validation is not None:
        validation_loader = DataLoader(
            validation,
            batch_size=batch_size,
            pin_memory=False,
            sampler=None,
            shuffle=False
        )
    else:
        validation_loader = None
    loss_function = nn.MSELoss()
    autoencoder.train()
    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step()
        data_iterator = tqdm(
            dataloader,
            leave=True,
            unit='batch',
            postfix={
                'epo': epoch,
                'acc': '%.4f' % 0.0,
                'lss': '%.6f' % 0.0,
                'vls': '%.6f' % -1,
                'vac': '%.6f' % -1,
            },
            disable=silent,
        )
        for index, batch in enumerate(data_iterator):
            # unpack the batch if its consists of a (feature, prediction) tuple or list
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, _ = batch  # if we have a prediction label, strip it away
            if cuda:
                batch = batch.cuda(async=True)
            batch = batch.squeeze(1).view(batch.size(0), -1)
            # run the batch through the autoencoder and obtain the output
            if corruption is not None:
                output = autoencoder(F.dropout(batch, corruption))
            else:
                output = autoencoder(batch)
            loss = loss_function(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            if update_freq is not None and index % update_freq == 0:
                accuracy = pretrain_accuracy(output, batch)  # possibly not the most useful metric
                loss_value = float(loss.item())
                # if we have validation data, then calculate the more useful validation loss
                if validation_loader is not None:
                    validation_output = predict(
                        validation,
                        autoencoder,
                        batch_size,
                        cuda=cuda,
                        silent=True,
                        encode=False
                    )
                    validation_inputs = []
                    for val_batch in validation_loader:
                        if (isinstance(val_batch, tuple) or isinstance(val_batch, list)) and len(val_batch) == 2:
                            validation_inputs.append(val_batch[0])
                        else:
                            validation_inputs.append(val_batch)
                    validation_actual = torch.cat(validation_inputs)
                    validation_loss = loss_function(validation_output, validation_actual)
                    validation_accuracy = pretrain_accuracy(validation_output, validation_actual)
                    validation_loss_value = float(validation_loss.item())
                    data_iterator.set_postfix(
                        epo=epoch,
                        acc='%.4f' % accuracy,
                        lss='%.6f' % loss_value,
                        vls='%.6f' % validation_loss_value,
                        vac='%.6f' % validation_accuracy
                    )
                    autoencoder.train()
                else:
                    validation_loss_value = -1
                    validation_accuracy = -1
                    data_iterator.set_postfix(
                        epo=epoch,
                        acc='%.4f' % accuracy,
                        lss='%.6f' % loss_value,
                        vls='%.6f' % -1,
                        vac='%.6f' % -1
                    )
                if update_callback is not None:
                    update_callback(accuracy, loss_value, validation_accuracy, validation_loss_value)
        if epoch_callback is not None:
            autoencoder.eval()
            epoch_callback(epoch, autoencoder)
            autoencoder.train()


def pretrain(dataset,
             autoencoder: StackedDenoisingAutoEncoder,
             epochs: int,
             batch_size: int,
             optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
             scheduler: Optional[Callable[[torch.nn.Module], Any]] = None,
             validation: Optional[torch.utils.data.Dataset] = None,
             corruption: Optional[float] = None,
             cuda: bool = True,
             sampler: Optional[torch.utils.data.sampler.Sampler] = None,
             silent: bool = False,
             update_freq: Optional[int] = None,
             update_callback: Optional[Callable[[float, float], None]] = None,
             epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None) -> None:
    """
    Given an autoencoder, train it using the data provided in the dataset; for simplicity the accuracy is reported only
    on the training dataset. If the training dataset is a 2-tuple or list of (feature, prediction), then the prediction
    is stripped away.

    :param dataset: instance of Dataset to use for training
    :param autoencoder: instance of an autoencoder to train
    :param epochs: number of training epochs
    :param batch_size: batch size for training
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param optimizer: function taking model and returning optimizer
    :param scheduler: function taking optimizer and returning scheduler, or None to disable
    :param validation: instance of Dataset to use for validation
    :param cuda: whether CUDA is used, defaults to True
    :param sampler: sampler to use in the DataLoader, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, None disables, default 1/10 of epochs
    :param update_callback: function of accuracy and loss to update
    :param epoch_callback: function of epoch and model
    :return: None
    """
    current_dataset = dataset
    number_of_subautoencoders = len(autoencoder.dimensions) - 1
    for index in range(number_of_subautoencoders):
        encoder, decoder = autoencoder.get_stack(index)
        embedding_dimension = autoencoder.dimensions[index]
        hidden_dimension = autoencoder.dimensions[index+1]
        # manual override to prevent corruption for the last subautoencoder
        if index == (number_of_subautoencoders - 1):
            corruption = None
        # initialise the subautoencoder
        sub_autoencoder = DenoisingAutoencoder(
            embedding_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            activation=torch.nn.ReLU() if index != (number_of_subautoencoders - 1) else None,
            corruption=nn.Dropout(corruption) if corruption is not None else None,
        )
        if cuda:
            sub_autoencoder = sub_autoencoder.cuda()
        ae_optimizer = optimizer(sub_autoencoder)
        ae_scheduler = scheduler(ae_optimizer) if scheduler is not None else scheduler
        train(
            current_dataset,
            sub_autoencoder,
            epochs,
            batch_size,
            ae_optimizer,
            validation=validation,
            corruption=None,  # already have dropout in the DAE
            scheduler=ae_scheduler,
            cuda=cuda,
            sampler=sampler,
            silent=silent,
            update_freq=update_freq,
            update_callback=update_callback,
            epoch_callback=epoch_callback
        )
        # copy the weights
        sub_autoencoder.copy_weights(encoder, decoder)
        # pass the dataset through the encoder part of the subautoencoder
        if index != (number_of_subautoencoders - 1):
            current_dataset = SimpleDataset(
                predict(
                    current_dataset,
                    sub_autoencoder,
                    batch_size,
                    cuda=cuda,
                    silent=silent
                )
            )
        else:
            current_dataset = None  # minor optimisation on the last subautoencoder


def predict(
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        batch_size: int,
        cuda: bool = True,
        silent: bool = False,
        encode: bool = True) -> torch.Tensor:
    """
    Given a dataset, run the model in evaluation mode with the inputs in batches and concatenate the
    output.

    :param dataset: evaluation Dataset
    :param model: autoencoder for prediction
    :param batch_size: batch size
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param encode: whether to encode or use the full autoencoder
    :return: predicted features from the Dataset
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False
    )
    data_iterator = tqdm(
        dataloader,
        leave=False,
        unit='batch',
        disable=silent,
    )
    features = []
    if isinstance(model, torch.nn.Module):
        model.eval()
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label strip it away
        if cuda:
            batch = batch.cuda(async=True)
        batch = batch.squeeze(1).view(batch.size(0), -1)
        if encode:
            output = model.encode(batch)
        else:
            output = model(batch)
        features.append(output.detach().cpu())  # move to the CPU to prevent out of memory on the GPU
    return torch.cat(features)
