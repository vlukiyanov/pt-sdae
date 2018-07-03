import torch
from torch.utils.data import Dataset
from typing import List


class SimpleDataset(Dataset):
    def __init__(self, data: List[torch.Tensor]):
        self.data = data

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def pretrain_accuracy(output: torch.Tensor, batch: torch.Tensor) -> float:
    """
    Simple function reporting accuracy for the pre-training step; this just counts the number of
    values which exactly match and divides by the tensor size. This isn't a useful metric.

    :param output: [batch size, feature size] Tensor of dtype float
    :param batch: [batch size, features size] Tensor of dtype float
    :return: floating point accuracy in the range [0,1]
    """
    numerator = float((output == batch).view(-1).long().sum().data.cpu())
    denominator = float(output.view(-1).size()[0])
    return numerator / denominator
