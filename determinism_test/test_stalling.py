import torch
import torch.nn as nn


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


def f(params):
    module = Net()
    # module.share_memory()
    module.load_state_dict(
        {k: torch.from_numpy(v) for k, v in zip(module.state_dict().keys(), params)}
    )
    return


def test_stalling():
    params2 = [tensor.cpu().numpy() for tensor in Net().state_dict().values()]
    import torch.multiprocessing as mp

    with mp.Pool(1) as p:
        # apply_async_dill(p,f,(params2,))
        p.apply_async(f, (params2,)).get()


if __name__ == "__main__":
    test_stalling()

    Net().load_state_dict(Net().state_dict())

    test_stalling()
