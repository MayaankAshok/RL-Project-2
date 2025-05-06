import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np
import gym
from Env import UAVEnv
# from uav_gym_env import UAVGymEnv

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.actor_mu = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log std dev
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        mu = self.actor_mu(x)
        std = self.actor_log_std.exp().expand_as(mu)
        value = self.critic(x)
        return mu, std, value

def select_action(mu, std):
    dist = torch.distributions.Normal(mu, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum()
    return action.numpy(), log_prob, dist.entropy().sum()

def worker_func(global_net, optimizer, global_ep, global_ep_r, res_queue, idx, max_ep=100): #### 3000
    env = UAVEnv()
    # env = UAVGymEnv()

    state_dim = env.get_observation_space_info()['shape'][0]
    action_dim = env.get_action_space_info()['shape'][0]
    local_net = ActorCritic(state_dim, action_dim)
    local_net.load_state_dict(global_net.state_dict())

    for ep in range(max_ep):
        state = env.reset()
        # state, _ = env.reset()
        buffer_s, buffer_a, buffer_r, buffer_log_prob = [], [], [], []
        ep_r = 0
        while True:
            state_t = torch.tensor(state, dtype=torch.float32)
            mu, std, value = local_net(state_t)
            action, log_prob, entropy = select_action(mu, std)
            next_state, reward, done, _ = env.step(action)
            ep_r += reward

            buffer_s.append(state)
            buffer_a.append(action)
            buffer_r.append(reward)
            buffer_log_prob.append(log_prob)

            state = next_state

            if done or len(buffer_r) >= 20:
                next_state_t = torch.tensor(next_state, dtype=torch.float32)
                _, _, next_value = local_net(next_state_t)
                R = 0 if done else next_value.item()
                discounted_r = []
                for r in reversed(buffer_r):
                    R = r + 0.99 * R
                    discounted_r.insert(0, R)

                s_batch = torch.tensor(buffer_s, dtype=torch.float32)
                a_batch = torch.tensor(buffer_a, dtype=torch.float32)
                r_batch = torch.tensor(discounted_r, dtype=torch.float32)
                log_probs = torch.stack(buffer_log_prob)

                mu, std, values = local_net(s_batch)
                advantage = r_batch.unsqueeze(1) - values

                actor_loss = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                total_loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                total_loss.backward()
                for global_param, local_param in zip(global_net.parameters(), local_net.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_net.load_state_dict(global_net.state_dict())

                buffer_s, buffer_a, buffer_r, buffer_log_prob = [], [], [], []

            if done:
                print(f"Worker {idx}, Episode {ep}, Reward: {ep_r}")
                global_ep.value += 1
                global_ep_r.value = max(global_ep_r.value, ep_r)
                # res_queue.put(ep_r)    ###################
                if env._passed_through_window():
                    event_type = 'success'
                elif env._hit_wall():
                    event_type = 'crash'
                else:
                    event_type = 'timeout'

                res_queue.put((ep_r, event_type))
                break

