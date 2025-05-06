import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.apply(weights_init)

    def forward(self, state):
        x = self.net(state)
        mu = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action_scaled = self.action_low + (action + 1) * 0.5 * (self.action_high - self.action_low)
        return action_scaled, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        def q_net():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1)
            )
        self.q1 = q_net()
        self.q2 = q_net()
        self.apply(weights_init)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)
    def push(self, *transition):
        self.buffer.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(np.array(actions), dtype=torch.float32, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        )
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, env, gamma=0.99, tau=0.005, lr=1e-4, init_alpha=0.2):
        self.env = env
        obs_dim = env.get_observation_space_info()["shape"][0]
        act_dim = env.get_action_space_info()["shape"][0]
        low, high = env.get_action_space_info()["low"], env.get_action_space_info()["high"]

        self.actor = Actor(obs_dim, act_dim, low, high).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.target_critic = Critic(obs_dim, act_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        #? entropy coef
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=device)
        self.target_entropy = -act_dim

        #? optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy().flatten()

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        s, a, r, ns, d = self.replay_buffer.sample(batch_size)

        #? Critic update
        with torch.no_grad():
            na, nlogp = self.actor.sample(ns)
            q1_t, q2_t = self.target_critic(ns, na)
            q_t = torch.min(q1_t, q2_t) - torch.exp(self.log_alpha) * nlogp
            y = r + (1 - d) * self.gamma * q_t
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        #? Actor update
        a_new, logp = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, a_new)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (torch.exp(self.log_alpha) * logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        #? Alpha update
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        #? sync target networks
        for p, p_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'log_alpha': self.log_alpha
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.alpha_opt.load_state_dict(ckpt['alpha_opt'])
        self.log_alpha = ckpt['log_alpha']
