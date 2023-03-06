import random
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr.common.typing import Metrics
from torch.utils.data import DataLoader
from util import seed_everything


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class FlowerNumpyClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device=torch.device("cpu"),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.device = device
        self.seed = seed

    def get_parameters(self, config):
        return parameters_from_state_dict(self.model.state_dict())

    def fit(self, parameters, config):
        self.model.load_state_dict(state_dict_from_parameters(parameters, self.model))
        self._train(config)
        return (
            parameters_from_state_dict(self.model.state_dict()),
            len(self.train_loader.dataset),
            {},
        )

    def _update_seed(self):
        seed_everything(self.seed)
        self.seed = random.randint(0, 100000)

    def _train(self, config):
        self.model.train()
        self._update_seed()
        optimizer = optim.Adadelta(self.model.parameters(), lr=config["lr"])
        for epoch in range(config["epochs"]):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

    def evaluate(self, parameters, config):
        self.model.load_state_dict(state_dict_from_parameters(parameters, self.model))
        self._update_seed()
        loss, accuracy = self._evaluate(config)
        return loss, len(self.val_loader.dataset), {"accuracy": accuracy}

    @torch.no_grad()
    def _evaluate(self, config):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.view_as(pred)).sum().item()
        test_loss /= len(self.val_loader.dataset)
        accuracy = correct / len(self.val_loader.dataset)
        return test_loss, accuracy


def parameters_from_state_dict(state_dict: Dict[str, Any]) -> List[np.ndarray]:
    return [tensor.cpu().numpy() for tensor in state_dict.values()]


def state_dict_from_parameters(
    parameters: List[np.ndarray], module: nn.Module
) -> Dict[str, Any]:
    return {
        k: torch.from_numpy(v) for k, v in zip(module.state_dict().keys(), parameters)
    }


def get_client_generator(train_dataloaders, val_dataloaders):
    def get_client_from_cid(cid: str) -> fl.client.NumPyClient:
        assert int(cid) < len(train_dataloaders), "Client ID out of range"
        assert int(cid) < len(val_dataloaders), "Client ID out of range"
        seed_everything(int(cid))
        return FlowerNumpyClient(
            cid,
            Net(),
            train_dataloaders[int(cid)],
            val_dataloaders[int(cid)],
            seed=int(cid),
        )

    return get_client_from_cid


def weighted_average_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
