import torch.nn as nn
import torch
import numpy as np
from rich.panel import Panel

from myterial import green_light, green, teal, teal_light, grey


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class BaseActor(nn.Module):
    n_actions = 3  # P, N_r, N_l
    n_inputs = 4  # Rho, Phy, V, Omega
    n_units = (32, 32)  # n units in hidden layer

    def __init__(self):
        super(BaseActor, self).__init__()


class Actor(BaseActor):
    def __init__(self):
        """
            Actor (policy) model
        """
        super(Actor, self).__init__()

        # build model
        self.fc1 = nn.Linear(self.n_inputs, self.n_units[0])
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.n_units[0])
        self.fc2 = nn.Linear(self.n_units[0], self.n_units[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.n_units[1], self.n_actions)
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = self.relu1(self.fc1(state))
        x = self.bn1(x)
        x = self.relu2(self.fc2(x))
        return self.tanh(self.fc3(x))

    def __rich_console__(self, *args, **kwargs):
        yield Panel(
            f"[{grey}]{self}", title=f"[{green_light}]Actor", style=green
        )


class Critic(BaseActor):
    def __init__(self):
        """Critic (Value) Model."""
        super(Critic, self).__init__()

        # build model
        self.fcs1 = nn.Linear(self.n_inputs, self.n_units[0])
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.n_units[0])
        self.fc2 = nn.Linear(self.n_units[0] + self.n_actions, self.n_units[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.n_units[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        xs = self.relu1(self.fcs1(state))
        xs = self.bn1(xs)
        x = torch.cat((xs, action), dim=1)
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

    def __rich_console__(self, *args, **kwargs):
        yield Panel(
            f"[{grey}]{self}", title=f"[{teal_light}]Critic", style=teal
        )
