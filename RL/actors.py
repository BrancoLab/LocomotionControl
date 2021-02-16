import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


class BaseActor(nn.Module):
    n_actions = 3  # P, N_r, N_l
    n_inputs = 8  # Rho, Phy, dv, do, v, o, taur, tauls
    n_units = (1024, 1024)  # n units in hidden layer

    def __init__(self):
        super(BaseActor, self).__init__()


class Critic(BaseActor):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(
            self.n_inputs + self.n_actions, self.n_units[0]
        )
        self.linear2 = nn.Linear(self.n_units[0], self.n_units[1])
        self.linear3 = nn.Linear(self.n_units[1], self.n_actions)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(BaseActor):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(self.n_inputs, self.n_units[0])
        self.linear2 = nn.Linear(self.n_units[0], self.n_units[1])
        self.linear3 = nn.Linear(self.n_units[1], self.n_actions)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x
