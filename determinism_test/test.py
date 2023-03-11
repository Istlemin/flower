import random

import flwr as fl
import numpy as np
import torch
from client import get_client_generator, weighted_average_accuracy
from dataset import partition_dataset
from flwr.server.strategy import FedAvg
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_dataset = MNIST("./mnist", train=True, download=True, transform=transform)
val_dataset = MNIST("./mnist", train=False, transform=transform)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def seed_everything(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def complete_run(seed=None):
    if seed is not None:
        seed_everything(seed)

    num_clients = 5
    train_datasets = partition_dataset(train_dataset, num_clients)
    val_datasets = partition_dataset(val_dataset, num_clients)
    train_dataloaders = [
        torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        for dataset in train_datasets
    ]
    val_dataloaders = [
        torch.utils.data.DataLoader(dataset, batch_size=16) for dataset in val_datasets
    ]
    client_resources = None
    client_fn = get_client_generator(train_dataloaders, val_dataloaders)
    client_config = {
        "lr": 0.05,
        "epochs": 1,
    }
    strategy = FedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        fraction_fit=0.25,
        fraction_evaluate=0.25,
        on_fit_config_fn=lambda _: client_config,
        on_evaluate_config_fn=lambda _: client_config,
        evaluate_metrics_aggregation_fn=weighted_average_accuracy,
    )

    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        seed_fn=seed_everything if seed is not None else None,
        seed=seed,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        client_resources=client_resources,
        strategy=strategy,
    )

    return hist


if __name__ == "__main__":
    run1 = complete_run(0)
