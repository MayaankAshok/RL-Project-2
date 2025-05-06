import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
from collections import deque


class ActorCritic(nn.Module):
    """
    Combined actor-critic network
    """
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        features = self.shared(x)
        
        # Get action probabilities
        action_probs = F.softmax(self.actor(features), dim=-1)
        
        # Get state value
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def get_action(self, state, device):
        """
        Sample an action from the policy distribution
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, _ = self.forward(state)
        
        # Create a categorical distribution over action probabilities
        dist = Categorical(action_probs)
        
        # Sample an action
        action = dist.sample()
        
        # Return action and log probability
        return action.item(), dist.log_prob(action)


class A2CAgent:
    def __init__(self, env_name, n_envs=8, learning_rate=0.001, gamma=0.99, 
                 entropy_coef=0.01, value_coef=0.5, use_gae=True, gae_lambda=0.95,
                 max_grad_norm=0.5, log_dir="./logs"):
        """
        Initialize A2C agent with parallel environments
        
        Args:
            env_name: The name of the Gym environment
            n_envs: Number of parallel environments
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            entropy_coef: Coefficient for entropy loss
            value_coef: Coefficient for value loss
            use_gae: Whether to use Generalized Advantage Estimation
            gae_lambda: Lambda parameter for GAE
            max_grad_norm: Maximum norm for gradient clipping
            log_dir: Directory for TensorBoard logs
        """
        # Create vectorized environments
        self.env_name = env_name
        self.n_envs = n_envs
        self.envs = [gym.make(env_name) for _ in range(n_envs)]
        
        # A2C hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        
        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir)
        self.log_interval = 10
        
        # Track metrics
        self.episode_rewards = [0] * n_envs
        self.total_steps = 0
        self.updates = 0
        self.episodes_completed = 0
        self.running_reward = deque(maxlen=100)
        
        # Determine device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize the actor-critic network using the first environment as reference
        input_dim = self.envs[0].observation_space.shape[0]
        n_actions = self.envs[0].action_space.n
        self.model = ActorCritic(input_dim, n_actions).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        """
        Compute returns and advantages using GAE if enabled
        
        Args:
            rewards: List of rewards for each step [n_steps, n_envs]
            values: List of value estimates for each step [n_steps, n_envs]
            dones: List of done flags for each step [n_steps, n_envs]
            next_value: Value estimates for the state after the last step [n_envs]
            
        Returns:
            returns: Discounted returns [n_steps * n_envs]
            advantages: Advantage estimates [n_steps * n_envs]
        """
        returns = []
        advantages = []
        
        if self.use_gae:
            # Initialize advantage for the last state
            gae = 0
            
            # For each environment, compute GAE in reverse order
            for env_idx in range(self.n_envs):
                env_returns = []
                env_advantages = []
                
                next_value_env = next_value[env_idx].item()
                last_value = next_value_env
                
                # Process steps in reverse order
                for t in reversed(range(len(rewards))):
                    # If episode terminated, use 0 as the next value
                    if dones[t][env_idx]:
                        next_value_env = 0
                        
                    # TD error: r_t + γV(s_{t+1}) - V(s_t)
                    delta = rewards[t][env_idx] + self.gamma * next_value_env * (1 - dones[t][env_idx]) - values[t][env_idx].item()
                    
                    # Compute GAE: A_t = δ_t + (γλ)A_{t+1}
                    gae = delta + self.gamma * self.gae_lambda * (1 - dones[t][env_idx]) * gae
                    
                    # Insert at the beginning (we're going backwards)
                    env_advantages.insert(0, gae)
                    
                    # The value function targets (returns) are advantages + value estimates
                    env_returns.insert(0, gae + values[t][env_idx].item())
                    
                    # Update next value
                    next_value_env = values[t][env_idx].item()
                
                # Add this environment's returns and advantages to the main lists
                returns.extend(env_returns)
                advantages.extend(env_advantages)
        else:
            # Traditional n-step returns calculation
            for env_idx in range(self.n_envs):
                env_returns = []
                
                # Initialize with value of the state after the last step
                R = next_value[env_idx].item() * (1 - dones[-1][env_idx])
                
                # Compute returns backward
                for t in reversed(range(len(rewards))):
                    R = rewards[t][env_idx] + self.gamma * R * (1 - dones[t][env_idx])
                    env_returns.insert(0, R)
                
                returns.extend(env_returns)
            
            # Convert returns to tensor
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            
            # Flatten values for all steps and environments
            values_flat = torch.cat([values[t] for t in range(len(values))])
            
            # Compute advantages as returns - values
            advantages = returns_tensor - values_flat
        
        # Convert to tensors
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages (reduces variance)
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        return returns_tensor, advantages_tensor
            
flat)
            
            # Calculate entropy loss (to encourage exploration)
            entropy_loss = -entropies_flat.mean()
            
            # Combine losses
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update network parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Print statistics
            if (update + 1) % 10 == 0 and all_episode_rewards:
                avg_reward = np.mean(all_episode_rewards[-100:]) if len(all_episode_rewards) > 0 else 0
                print(f"Update {update+1}, Avg Reward (last 100 episodes): {avg_reward:.2f}, Episodes: {len(all_episode_rewards)}")
        
        # Close all environments
        for env in self.envs:
            env.close()
            
        return all_episode_rewards
    
    def evaluate(self, n_episodes=10, render=False):
        """
        Evaluate the trained agent
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment
        """
        # Create a separate environment for evaluation
        eval_env = gym.make(self.env_name)
        total_rewards = []
        
        for episode in range(n_episodes):
            state = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if render:
                    eval_env.render()
                
                # Get action from policy (use greedy policy for evaluation)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, _ = self.model(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()
                
                # Take action in environment
                state, reward, done, _ = eval_env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode+1}, Reward: {episode_reward:.2f}")
        
        print(f"Average Evaluation Reward: {np.mean(total_rewards):.2f}")
        eval_env.close()
        return total_rewards


# Example usage
if __name__ == "__main__":
    # Initialize and train A2C agent with parallel environments
    agent = A2CAgent(
        env_name='CartPole-v1',
        n_envs=16,              # Number of parallel environments
        learning_rate=0.001,
        gamma=0.99,
        entropy_coef=0.01,      # Entropy coefficient for exploration
        value_coef=0.5          # Value loss coefficient
    )
    
    # Train agent (n_updates = number of parameter updates, n_steps = steps per update)
    rewards = agent.train(n_updates=1000, n_steps=5)
    
    # Plot training curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('A2C Training Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig('a2c_training_curve.png')
    plt.show()
    
    # Evaluate agent
    eval_rewards = agent.evaluate(n_episodes=10, render=True)