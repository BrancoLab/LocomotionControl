import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import choices
from collections import namedtuple, deque
import copy

from pyrnn._utils import torchify, npify

from control import model
from control.config import dt, MOUSE
from control.utils import merge

from RL.actors import Actor, Critic
from RL import settings

memory = namedtuple("memory", "s, a, r, s_prime, done")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RLAgent(model.Model):
    def __init__(self):
        model.Model.__init__(self)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=settings.LR_ACTOR
        )

        # Critic Network (w/ Target Network)
        self.critic_local = Critic().to(device)
        self.critic_target = Critic().to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=settings.LR_CRITIC,
            weight_decay=settings.WEIGHT_DECAY,
        )

        self.memory = Memory()
        self.noise = OUNoise()

        self.loss = nn.MSELoss()

    def __rich_console__(self, *args, **kwargs):
        yield "A2A Agent"
        yield f"    memory: {len(self.memory)}\n"
        yield str(self.actor_local)
        yield "\n"
        yield str(self.critic_local)

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        """
            Choose controls
        """
        self.actor_local.eval()
        with torch.no_grad():
            action = npify(self.actor_local(torchify(state)))
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        action = action.T
        if np.any(np.isnan(action)):
            return None
        return np.clip(action, -1, 1)

    def move(self, controls):
        """
            Given selected controls, play out physics to move
        """
        u = model.control(*controls)
        x = model.state(*np.array(self.curr_x))

        variables = merge(u, x, MOUSE)
        inputs = [variables[a] for a in self._M_args]
        dxdt = self.calc_dxdt(*inputs).ravel()

        self.curr_x = model.state(*(np.array(self.curr_x) + dxdt * dt))

    def step(self, state, controls, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, controls, reward, next_state, done)

        # if have enough samples, do some learnin
        if len(self.memory) > settings.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (settings.GAMMA * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = self.loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

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
        # if len(self) > settings.BUFFER_SIZE:
        #     self.replay_memory.pop(0)

        self.mem.append(
            memory(state, controls.ravel(), reward, next_state, done)
        )

    def get_batch(self):
        return choices(self.mem, k=self.batch_size)

    def sample(self):
        batch = self.get_batch()

        S = torchify(np.vstack([m.s for m in batch])).float().to(device)
        A = torchify(np.vstack([m.a for m in batch])).float().to(device)
        R = torchify(np.vstack([m.r for m in batch])).float().to(device)
        Sp = torchify(np.vstack([m.s_prime for m in batch])).float().to(device)
        D = torchify(np.vstack([m.done for m in batch])).float().to(device)

        if len(S) != len(A):
            raise ValueError("bad memory")
        return (S, A, R, Sp, D)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size=3, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

        # Thanks to Hiu C. for this tip, this really helped get the learning up to the desired levels
        dx = self.theta * (
            self.mu - x
        ) + self.sigma * np.random.standard_normal(self.size)

        self.state = x + dx
        return self.state
