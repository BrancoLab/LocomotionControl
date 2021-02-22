import torch
import torch.nn as nn
import torch.autograd
import numpy as np


def fanin_(size):
    fan_in = size[0]
    weight = 1.0 / np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)


class BaseActor(nn.Module):
    n_actions = 3  # P, N_r, N_l
    n_inputs = 8  # Rho, Phy, dv, do, v, o, taur, tauls
    n_units = (512, 512)  # n units in hidden layer

    def __init__(self):
        super(BaseActor, self).__init__()


class Critic(BaseActor):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(self.n_inputs, self.n_units[0])
        self.linear2 = nn.Linear(
            self.n_units[0] + self.n_actions, self.n_units[1]
        )
        self.linear3 = nn.Linear(self.n_units[1], self.n_units[1])
        self.linear4 = nn.Linear(self.n_units[1], self.n_units[1])
        self.linear5 = nn.Linear(self.n_units[1], self.n_actions)

        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        self.linear3.weight.data = fanin_(self.linear3.weight.data.size())
        self.linear4.weight.data = fanin_(self.linear4.weight.data.size())
        self.linear5.weight.data.uniform_(-3e-3, 3e-3)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = self.relu(self.linear1(state))
        x = torch.cat([x, action], 1)
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        return self.linear5(x)


class Actor(BaseActor):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(self.n_inputs, self.n_units[0])
        self.linear2 = nn.Linear(self.n_units[0], self.n_units[1])
        self.linear3 = nn.Linear(self.n_units[1], self.n_units[1])
        self.linear4 = nn.Linear(self.n_units[1], self.n_units[1])
        self.linear5 = nn.Linear(self.n_units[1], self.n_actions)

        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        self.linear3.weight.data = fanin_(self.linear3.weight.data.size())
        self.linear4.weight.data = fanin_(self.linear4.weight.data.size())
        self.linear5.weight.data.uniform_(-0.003, 0.003)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = self.relu(self.linear1(state))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = torch.tanh(self.linear5(x))
        return x
