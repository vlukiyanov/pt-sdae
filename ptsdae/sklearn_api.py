from typing import Optional, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from scipy.sparse import issparse


from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae


class SDAETransformer(TransformerMixin, BaseEstimator):
    def __init__(self,
                 dimensions: List[int],
                 cuda: Optional[bool] = None,
                 batch_size: int = 256,
                 pretrain_epochs: int = 200,
                 finetune_epochs: int = 500,
                 lr: float = 0.1,
                 corruption: Optional[float] = 0.2,
                 final_activation: Optional[torch.nn.Module] = None) -> None:
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.corruption = corruption
        self.autoencoder = None
        self.final_activation = final_activation

    def fit(self, X, y=None) -> None:
        if issparse(X):
            X = X.todense()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        self.autoencoder = StackedDenoisingAutoEncoder(self.dimensions, final_activation=self.final_activation)
        if self.cuda:
            self.autoencoder.cuda()
        ae.pretrain(
            ds,
            self.autoencoder,
            cuda=self.cuda,
            epochs=self.pretrain_epochs,
            batch_size=self.batch_size,
            optimizer=lambda model: SGD(model.parameters(), lr=self.lr, momentum=0.9),
            scheduler=lambda x: StepLR(x, 100, gamma=0.1),
            corruption=0.2,
            silent=True
        )
        ae_optimizer = SGD(params=self.autoencoder.parameters(), lr=self.lr, momentum=0.9)
        ae.train(
            ds,
            self.autoencoder,
            cuda=self.cuda,
            epochs=self.finetune_epochs,
            batch_size=self.batch_size,
            optimizer=ae_optimizer,
            scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
            corruption=self.corruption,
            silent=True
        )

    def transform(self, X):
        if self.autoencoder is None:
            raise NotFittedError
        if issparse(X):
            X = X.todense()
        self.autoencoder.eval()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        dataloader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False
        )
        features = []
        for index, batch in enumerate(dataloader):
            batch = batch[0]
            if self.cuda:
                batch = batch.cuda(non_blocking=True)
            features.append(self.autoencoder.encoder(batch).detach().cpu())
        return torch.cat(features).numpy()

    def score(self, X, y=None, sample_weight=None) -> float:
        loss_function = torch.nn.MSELoss()
        if self.autoencoder is None:
            raise NotFittedError
        if issparse(X):
            X = X.todense()
        self.autoencoder.eval()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        dataloader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False
        )
        loss = 0
        for index, batch in enumerate(dataloader):
            batch = batch[0]
            if self.cuda:
                batch = batch.cuda(non_blocking=True)
            output = self.autoencoder(batch)
            loss += float(loss_function(output, batch).item())
        return loss
