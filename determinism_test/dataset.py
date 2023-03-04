import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


def partition_dataset(dataset: Dataset, num_partitions: int):
    """Partition dataset into num_partitions partitions."""
    length = len(dataset)
    # shuffle dataset
    indices = torch.randperm(length).tolist()
    dataset = Subset(dataset, indices)
    partitions = []
    for i in range(num_partitions):
        start = i * length // num_partitions
        end = (i + 1) * length // num_partitions
        partitions.append(Subset(dataset, range(start, end)))
    return partitions
