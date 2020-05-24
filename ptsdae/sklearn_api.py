from typing import Optional, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from typing import Any, Callable
from scipy.sparse import issparse

from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae


class SDAETransformerBase(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        dimensions: List[int],
        cuda: Optional[bool] = None,
        batch_size: int = 256,
        pretrain_epochs: int = 200,
        finetune_epochs: int = 500,
        corruption: Optional[float] = 0.2,
        optimiser_pretrain: Callable[
            [torch.nn.Module], torch.optim.Optimizer
        ] = lambda x: SGD(x.parameters(), lr=0.1, momentum=0.9),
        optimiser_train: Callable[
            [torch.nn.Module], torch.optim.Optimizer
        ] = lambda x: SGD(x.parameters(), lr=0.1, momentum=0.9),
        scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = lambda x: StepLR(
            x, 100, gamma=0.1
        ),
        final_activation: Optional[torch.nn.Module] = None,
    ) -> None:
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.optimiser_pretrain = optimiser_pretrain
        self.optimiser_train = optimiser_train
        self.scheduler = scheduler
        self.corruption = corruption
        self.autoencoder = None
        self.final_activation = final_activation

    def fit(self, X, y=None):
        if issparse(X):
            X = X.todense()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        self.autoencoder = StackedDenoisingAutoEncoder(
            self.dimensions, final_activation=self.final_activation
        )
        if self.cuda:
            self.autoencoder.cuda()
        ae.pretrain(
            ds,
            self.autoencoder,
            cuda=self.cuda,
            epochs=self.pretrain_epochs,
            batch_size=self.batch_size,
            optimizer=self.optimiser_pretrain,
            scheduler=self.scheduler,
            corruption=0.2,
            silent=True,
        )
        ae_optimizer = self.optimiser_train(self.autoencoder)
        ae.train(
            ds,
            self.autoencoder,
            cuda=self.cuda,
            epochs=self.finetune_epochs,
            batch_size=self.batch_size,
            optimizer=ae_optimizer,
            scheduler=self.scheduler(ae_optimizer),
            corruption=self.corruption,
            silent=True,
        )
        return self

    def score(self, X, y=None, sample_weight=None) -> float:
        loss_function = torch.nn.MSELoss()
        if self.autoencoder is None:
            raise NotFittedError
        if issparse(X):
            X = X.todense()
        self.autoencoder.eval()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        loss = 0
        for index, batch in enumerate(dataloader):
            batch = batch[0]
            if self.cuda:
                batch = batch.cuda(non_blocking=True)
            output = self.autoencoder(batch)
            loss += float(loss_function(output, batch).item())
        return loss


def _transform(X, autoencoder, batch_size, cuda):
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    features = []
    for batch in dataloader:
        batch = batch[0]
        if cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(autoencoder.encoder(batch).detach().cpu())
    return torch.cat(features).numpy()


class SDAETransformer(SDAETransformerBase):
    def transform(self, X):
        if self.autoencoder is None:
            raise NotFittedError
        if issparse(X):
            X = X.todense()
        self.autoencoder.eval()
        return _transform(X, self.autoencoder, self.batch_size, self.cuda)


class SDAERepresentationTransformer(SDAETransformerBase):
    def transform(self, X):
        if self.autoencoder is None:
            raise NotFittedError
        if issparse(X):
            X = X.todense()
        self.autoencoder.eval()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        features_encoder = [[] for _ in self.autoencoder.encoder]
        features_decoder = [[] for _ in self.autoencoder.decoder]
        for index, batch in enumerate(dataloader):
            batch = batch[0]
            if self.cuda:
                batch = batch.cuda(non_blocking=True)
            for index, unit in enumerate(self.autoencoder.encoder):
                batch = unit(batch)
                features_encoder[index].append(batch.detach().cpu())
            for index, unit in enumerate(self.autoencoder.decoder):
                batch = unit(batch)
                features_decoder[index].append(batch.detach().cpu())
        return np.concatenate(
            [torch.cat(x).numpy() for x in features_encoder + features_decoder[:-1]],
            axis=1,
        )
