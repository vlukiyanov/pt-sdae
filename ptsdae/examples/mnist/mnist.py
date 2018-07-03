import click
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.cluster import KMeans

from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy


class CachedMNIST(Dataset):
    def __init__(self, train, cuda):
        img_transform = transforms.Compose([
            transforms.Lambda(self._transformation)
        ])
        self.ds = MNIST(
            './data',
            download=True,
            train=train,
            transform=img_transform
        )
        self.cuda = cuda
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float() * 0.02

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(async=True)
                self._cache[index][1] = self._cache[index][1].cuda(async=True)
        return self._cache[index]

    def __len__(self) -> int:
        return len(self.ds)


@click.command()
@click.option(
    '--cuda',
    help='whether to use CUDA (default False).',
    type=bool,
    default=False
)
@click.option(
    '--batch-size',
    help='training batch size (default 256).',
    type=int,
    default=256
)
@click.option(
    '--pretrain-epochs',
    help='number of pretraining epochs (default 300).',
    type=int,
    default=300
)
@click.option(
    '--finetune-epochs',
    help='number of finetune epochs (default 500).',
    type=int,
)
def main(
    cuda,
    batch_size,
    pretrain_epochs,
    finetune_epochs
):
    ds_train = CachedMNIST(train=True, cuda=cuda)
    ds_val = CachedMNIST(train=False, cuda=cuda)
    autoencoder = StackedDenoisingAutoEncoder(
        [28 * 28, 500, 500, 2000, 10],
        final_activation=None
    ).cuda()
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2
    )
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
    )
    dataloader = DataLoader(
        ds_train,
        batch_size=1024,
        shuffle=False
    )
    kmeans = KMeans(n_clusters=10, n_init=20)
    autoencoder.eval()
    features = []
    actual = []
    for index, batch in enumerate(dataloader):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value)
        if cuda:
            batch = batch.cuda(async=True)
        batch = batch.squeeze(1).view(batch.size(0), -1)
        features.append(autoencoder.encoder(batch).detach().cpu())
    actual = torch.cat(actual).long()
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
    print('Final k-Means accuracy: %s' % accuracy)


if __name__ == '__main__':
    main()
