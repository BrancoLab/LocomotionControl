import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import choices
from collections import namedtuple, deque

from pyrnn._utils import torchify, npify

from control import model
from control.config import dt, MOUSE
from control.utils import merge

from RL.actors import Actor, Critic
from RL import settings

memory = namedtuple("memory", "state, action, reward, next_state, done")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RLAgent(model.Model):
    prev_x = None

    def __init__(self):
        model.Model.__init__(self)

        # Actor Network (w/ Target Network)
        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)

        # Critic Network (w/ Target Network)
        self.critic = Critic().to(device)
        self.critic_target = Critic().to(device)

        # init networks with same params
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(param.data)

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(param.data)

        # optimizers
        self.criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=settings.LR_ACTOR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=settings.LR_CRITIC,
            weight_decay=settings.WEIGHT_DECAY,
        )

        # loss
        self.memory = Memory()
        self.noise = OUNoise()

    def __rich_console__(self, *args, **kwargs):
        yield "A2A Agent"
        yield f"    memory: {len(self.memory)}\n"
        yield str(self.actor)
        yield "\n"
        yield str(self.critic)

    def get_controls(self, state, frame, add_noise=True):
        """
            Choose controls
        """
        self.actor.eval()
        with torch.no_grad():
            action = npify(self.actor(torchify(state)))
        self.actor.train()

        if add_noise:
            # if np.random.random() < .4:
            action += np.random.normal(0, settings.NOISE_SCALE, size=3)
            # action += self.noise.sample(frame)
        action = action.T

        if np.any(np.isnan(action)):
            return None

        # return action
        return np.clip(action, -1, 1)

    def move(self, controls):
        """
            Given selected controls, play out physics to move
        """
        self.prev_x = self.curr_x
        u = model.control(*controls)
        x = model.state(*np.array(self.curr_x))

        variables = merge(u, x, MOUSE)
        inputs = [variables[a] for a in self._M_args]
        dxdt = self.calc_dxdt(*inputs).ravel()
        self.curr_dxdt = model._dxdt(*dxdt)

        self.curr_x = model.state(*(np.array(self.curr_x) + dxdt * dt))

    def fit(self):
        """
            Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        # if have enough samples, do some learnin
        if len(self.memory) <= settings.BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + settings.GAMMA * next_Q
        critic_loss = self.criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(
            states, self.actor.forward(states)
        ).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                settings.TAU * local_param.data
                + (1.0 - settings.TAU) * target_param.data
            )


class Memory:
    def __init__(self):
        self.mem = deque(maxlen=settings.BUFFER_SIZE)
        self.batch_size = settings.BATCH_SIZE

    def __len__(self):
        return len(self.mem)

    def add(self, state, controls, reward, next_state, done):
        self.mem.append(
            memory(state, controls.ravel(), reward, next_state, done)
        )

    def get_batch(self):
        return choices(self.mem, k=self.batch_size)

    def sample(self):
        batch = self.get_batch()

        S = torchify(np.vstack([m.state for m in batch])).float().to(device)
        A = torchify(np.vstack([m.action for m in batch])).float().to(device)
        R = torchify(np.vstack([m.reward for m in batch])).float().to(device)

        Sp = (
            torchify(np.vstack([m.next_state for m in batch]))
            .float()
            .to(device)
        )

        D = torchify(np.vstack([m.done for m in batch])).float().to(device)

        if len(S) != len(A):
            raise ValueError("bad memory")
        return (S, A, R, Sp, D)


class OUNoise(object):
    def __init__(
        self,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=100000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = 3
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(
            self.action_dim
        )
        self.state = x + dx
        return self.state

    def sample(self, t=0):
        noise = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return noise
