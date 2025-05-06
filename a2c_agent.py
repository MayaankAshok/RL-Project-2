import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from Env import UAVEnv
from parallel_env import SequentialEnvs  # Use sequential implementation instead of parallel
import time
import os

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network for A2C algorithm
    Actor: Policy network that outputs mean and std for each action dimension
    Critic: Value network that estimates the state value function
    """
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super(ActorCriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Store action bounds for scaling
        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)
        
        # Shared network layers - INCREASED SIZE FOR BETTER CAPACITY
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor network (Policy)
        self.actor_mean = nn.Linear(128, action_dim)
        # Initialize log_std to a larger negative value for more exploration
        self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        
        # Critic network (Value)
        self.critic = nn.Linear(128, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def forward(self, state):
        """Forward pass through network"""
        x = self.shared_layers(state)
        
        # Actor: output mean and std
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_log_std)
        
        # Critic: output state value
        value = self.critic(x)
        
        return action_mean, action_std, value
    
    def sample_action(self, state):
        """Sample an action from the policy distribution"""
        # Convert state to tensor and add batch dimension if necessary
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Get mean and std from policy without tracking gradients
        with torch.no_grad():
            action_mean, action_std, _ = self.forward(state)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        # Sample action from distribution
        action_raw = dist.sample()
        
        # Get log probability of action
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        
        # Scale action to environment range
        action = self.scale_action(action_raw)
        # action[0,2] = 1
        return action.cpu().numpy().flatten(), log_prob.cpu().item()
    
    def evaluate_action(self, state, action):
        """Evaluate log probability and entropy of a given action"""
        # Get mean and std from policy
        action_mean, action_std, value = self.forward(state)
        
        # Convert action back to raw scale (unscale)
        action_raw = self.unscale_action(action)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        # Get log probability of action
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        
        # Calculate entropy of policy
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value
    
    def scale_action(self, action_raw):
        """Scale action from [-1, 1] to environment action space"""
        # Tanh to restrict to [-1, 1]
        action_tanh = torch.tanh(action_raw)
        
        # Scale to environment range
        scaled_action = self.action_low + (self.action_high - self.action_low) * (action_tanh + 1) / 2
        
        return scaled_action
    
    def unscale_action(self, action):
        """Convert action from environment space to raw network output space"""
        # Reverse scaling from environment range to [-1, 1]
        normalized = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
        
        # Inverse of tanh (arctanh) to get raw network output
        # Clip to avoid numerical instability
        normalized = torch.clamp(normalized, -0.999, 0.999)
        raw_action = torch.atanh(normalized)
        
        return raw_action

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent
    """
    def __init__(self, env, checkpoint_dir="./checkpoints", num_envs=4):
        """
        Initialize A2C agent with support for sequential environment batching
        
        Args:
            env: Environment instance (only used for single-environment mode)
            checkpoint_dir: Directory to save/load checkpoints
            num_envs: Number of environments to run in batch
        """
        self.env = env
        self.num_envs = num_envs
        
        # For batch training, create multiple environments
        if num_envs > 1:
            env_fns = [lambda: UAVEnv(max_steps=500) for _ in range(num_envs)]
            self.parallel_env = SequentialEnvs(env_fns)  # Using sequential implementation
            
            # Use properties from first environment to set up network
            self.action_low = self.parallel_env.action_low
            self.action_high = self.parallel_env.action_high
            self.state_dim = 6  # 3D position + 3D velocity
            self.action_dim = 3  # 2 orientation angles + thrust
        else:
            self.parallel_env = None
            self.state_dim = 6  # 3D position + 3D velocity
            self.action_dim = 3  # 2 orientation angles + thrust
            self.action_low = env.action_low
            self.action_high = env.action_high
        
        # Create actor-critic network
        self.network = ActorCriticNetwork(
            self.state_dim, 
            self.action_dim, 
            self.action_low, 
            self.action_high
        ).to(device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        
        # Hyperparameters - ADJUSTED FOR BETTER LEARNING
        self.gamma = 0.99  # Discount factor
        self.entropy_coef = 0.05  # INCREASED for more exploration
        self.value_loss_coef = 0.5  # Value loss coefficient
        
        # For saving checkpoints
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def learn(self, num_episodes=1000, max_steps=1000, update_freq=10, 
              render=False, save_freq=100, log_freq=10):
        """Train the agent using A2C algorithm"""
        # Use parallel environment training if enabled
        if self.num_envs > 1:
            return self._learn_parallel(num_episodes, max_steps, update_freq, render, save_freq, log_freq)
        else:
            return self._learn_single(num_episodes, max_steps, update_freq, render, save_freq, log_freq)
            
    def _learn_parallel(self, num_episodes=1000, max_steps=1000, update_freq=10, 
                      render=False, save_freq=100, log_freq=10):
        """Train the agent using A2C algorithm with parallel environments"""
        # Training statistics
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        moving_avg_reward = []
        
        # Create figure for live plot
        if render:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Start training
        print(f"Starting A2C training with {self.num_envs} parallel environments...")
        
        # Reset environments to get initial states
        states = self.parallel_env.reset()
        
        # Initialize episode tracking for each env
        env_episode_rewards = [0] * self.num_envs
        env_episode_steps = [0] * self.num_envs
        total_episodes = 0
        
        # Main training loop
        while total_episodes < num_episodes:
            # Storage for environment interactions
            batch_states = []
            batch_actions = []
            batch_log_probs = []
            batch_rewards = []
            batch_dones = []
            
            # Collect data from environments
            for _ in range(update_freq):
                # Sample actions for all environments
                actions = []
                log_probs = []
                
                # Convert states to tensor
                states_tensor = torch.FloatTensor(states).to(device)
                
                # Get actions from policy
                with torch.no_grad():
                    # Process all states at once for efficiency
                    action_means, action_stds, _ = self.network(states_tensor)
                    
                    # Create normal distributions
                    dists = Normal(action_means, action_stds)
                    
                    # Sample actions
                    action_raws = dists.sample()
                    
                    # Get log probabilities
                    log_prob_batch = dists.log_prob(action_raws).sum(dim=-1)
                    
                    # Scale actions to environment range
                    actions_tensor = self.network.scale_action(action_raws)
                    
                    # Convert to numpy
                    actions = actions_tensor.cpu().numpy()
                    log_probs = log_prob_batch.cpu().numpy()
                
                # Take step in all environments
                next_states, rewards, dones, infos = self.parallel_env.step(actions)
                
                # Store data
                batch_states.append(states)
                batch_actions.append(actions)
                batch_log_probs.append(log_probs)
                batch_rewards.append(rewards)
                batch_dones.append(dones)
                
                # Update episode tracking
                for env_idx in range(self.num_envs):
                    env_episode_rewards[env_idx] += rewards[env_idx]
                    env_episode_steps[env_idx] += 1
                
                # Process episodes that are done
                for env_idx in range(self.num_envs):
                    if dones[env_idx]:
                        # Store completed episode's stats
                        episode_rewards.append(env_episode_rewards[env_idx])
                        episode_lengths.append(env_episode_steps[env_idx])
                        
                        # Check if episode was successful
                        # Use the correct hit_wall method from the environment instance
                        pos_y = infos[env_idx].get("position", [0, -1, 0])[1]
                        # An episode is successful if it ended on positive y side AND didn't hit the wall
                        success = pos_y > 0 and not self.parallel_env.envs[env_idx]._hit_wall()
                        
                        episode_success.append(int(success))  # Store as 0 or 1 for proper averaging
                        
                        # Reset this environment's tracking
                        env_episode_rewards[env_idx] = 0
                        env_episode_steps[env_idx] = 0
                        total_episodes += 1
                        
                        # Reset the environment that finished an episode
                        states[env_idx] = self.parallel_env.reset_one(env_idx)
                
                # Update current states
                states = next_states
                
                # Optional rendering (renders only the first environment)
                if render:
                    self.parallel_env.render(0)
                    plt.pause(0.01)
            
            # Convert batch data to numpy arrays
            b_states = np.array(batch_states).reshape(-1, self.state_dim)
            b_actions = np.array(batch_actions).reshape(-1, self.action_dim)
            b_log_probs = np.array(batch_log_probs).flatten()
            b_rewards = np.array(batch_rewards).flatten()
            b_dones = np.array(batch_dones).flatten()
            
            # Update policy with collected batch data
            self._update_policy_parallel(b_states, b_actions, b_log_probs, b_rewards, b_dones)
            
            # Calculate moving average of rewards
            if len(episode_rewards) > 10:
                avg = np.mean(episode_rewards[-10:])
            else:
                avg = np.mean(episode_rewards) if episode_rewards else 0
            moving_avg_reward.append(avg)
            
            # Log progress
            if total_episodes % log_freq == 0 and episode_rewards:
                # Calculate success rate over recent episodes
                if len(episode_success) >= 10:
                    success_rate = np.mean(episode_success[-10:])  # Will be value between 0 and 1
                else:
                    success_rate = np.mean(episode_success) if episode_success else 0
                
                # Count number of successes in the most recent batch
                recent_successes = sum(episode_success[-10:]) if len(episode_success) >= 10 else sum(episode_success)
                recent_total = min(10, len(episode_success))
                
                print(f"Episodes: {total_episodes}/{num_episodes}, " +
                      f"Avg Reward: {avg:.1f}, Success Rate: {success_rate:.2%} " +
                      f"({recent_successes}/{recent_total} recent episodes)")
                
                # Update live plot
                if render:
                    self._update_plot(ax1, ax2, episode_rewards, moving_avg_reward, episode_success)
            
            # Save checkpoint
            if total_episodes % save_freq == 0 and total_episodes > 0:
                self.save_checkpoint(total_episodes)
        
        # End of training
        print("Training completed!")
        
        # Save final model
        self.save_checkpoint("final")
        
        # Save training results
        np.save(f"{self.checkpoint_dir}/episode_rewards.npy", np.array(episode_rewards))
        np.save(f"{self.checkpoint_dir}/episode_lengths.npy", np.array(episode_lengths))
        np.save(f"{self.checkpoint_dir}/episode_success.npy", np.array(episode_success))
        
        # Final plot
        if render:
            plt.ioff()
            self._plot_training_results(episode_rewards, moving_avg_reward, episode_success)
        
        # Clean up parallel environments
        self.parallel_env.close()
        
        return episode_rewards, episode_lengths, episode_success
    
    def _update_policy_parallel(self, states, actions, log_probs_old, rewards, dones):
        """Update policy using data from parallel environments"""
        if len(states) == 0:
            return
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.FloatTensor(actions).to(device)
        
        # Compute discounted returns and advantages
        returns = self._compute_returns_parallel(rewards, dones)
        returns_tensor = torch.FloatTensor(returns).to(device)
        
        # Forward pass through network
        log_probs, entropies, values = self.network.evaluate_action(states_tensor, actions_tensor)
        values = values.squeeze()
        
        # Compute advantage estimates
        advantages = returns_tensor - values.detach()
        
        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Handle scalar vs batch case for critic loss to prevent broadcasting warning
        if returns_tensor.dim() == 0 or returns_tensor.size(0) == 1:
            # For single samples, use direct calculation
            critic_loss = (values - returns_tensor).pow(2).mean()
        else:
            # For batches, use MSE loss
            critic_loss = F.mse_loss(values, returns_tensor)
            
        entropy_loss = -entropies.mean()
        
        # Total loss
        loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
    
    def _compute_returns_parallel(self, rewards, dones):
        """Compute returns for batch data from parallel environments"""
        batch_size = len(rewards)
        returns = np.zeros_like(rewards)
        
        # We need to handle potentially multiple episode fragments in the batch
        R = 0
        for i in reversed(range(batch_size)):
            # If episode ended, reset return calculation
            if dones[i]:
                R = 0
            
            # Update return with current reward and future return
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns[i] = R
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            returns = np.clip(returns, -10.0, 10.0)
        
        return returns

    def _learn_single(self, num_episodes=1000, max_steps=1000, update_freq=10, 
                     render=False, save_freq=100, log_freq=10):
        """Train the agent using A2C algorithm with a single environment"""
        # Training statistics
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        moving_avg_reward = []
        
        # Create figure for live plot
        if render:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Start training
        print("Starting A2C training...")
        
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # Storage for episode data
            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []
            
            # Run episode
            while not done and steps < max_steps:
                # Get action from policy
                action, log_prob = self.network.sample_action(state)
                
                # Take step in environment
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                
                # Update state
                state = next_state
                total_reward += reward
                steps += 1
                
                # Render environment if requested
                if render:
                    self.env.render()
                    plt.pause(0.01)
                
                # Update policy if enough steps have been taken
                if steps % update_freq == 0 or done:
                    self._update_policy(states, actions, log_probs, rewards, dones)
                    
                    # Clear episode data
                    states = []
                    actions = []
                    log_probs = []
                    rewards = []
                    dones = []
            
            # End of episode
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Check if episode was successful
            success = (self.env.state[1] > 0 and not self.env._hit_wall())
            episode_success.append(success)
            
            # Calculate moving average
            if len(episode_rewards) > 10:
                avg = np.mean(episode_rewards[-10:])
            else:
                avg = np.mean(episode_rewards)
            moving_avg_reward.append(avg)
            
            # Log progress
            if episode % log_freq == 0:
                success_rate = np.mean(episode_success[-10:]) if len(episode_success) >= 10 else np.mean(episode_success)
                print(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.1f}, " +
                      f"Steps: {steps}, Moving Avg: {avg:.1f}, Success Rate: {success_rate:.2%}")
                
                # Update live plot
                if render:
                    self._update_plot(ax1, ax2, episode_rewards, moving_avg_reward, episode_success)
            
            # Save checkpoint
            if episode % save_freq == 0:
                self.save_checkpoint(episode)
        
        # End of training
        print("Training completed!")
        
        # Save final model
        self.save_checkpoint("final")
        
        # Save training results
        np.save(f"{self.checkpoint_dir}/episode_rewards.npy", np.array(episode_rewards))
        np.save(f"{self.checkpoint_dir}/episode_lengths.npy", np.array(episode_lengths))
        np.save(f"{self.checkpoint_dir}/episode_success.npy", np.array(episode_success))
        
        # Final plot
        if render:
            plt.ioff()
            self._plot_training_results(episode_rewards, moving_avg_reward, episode_success)
        
        return episode_rewards, episode_lengths, episode_success

    def _update_policy(self, states, actions, log_probs_old, rewards, dones):
        """Update policy using collected trajectory data"""
        if len(states) == 0:
            return
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(device)
        
        # Compute discounted returns and advantages
        returns = self._compute_returns(rewards, dones)
        returns_tensor = torch.FloatTensor(returns).to(device)
        
        # Forward pass through network
        log_probs, entropies, values = self.network.evaluate_action(states_tensor, actions_tensor)
        values = values.squeeze()
        
        # Compute advantage estimates
        advantages = returns_tensor - values.detach()
        
        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Handle scalar vs batch case for critic loss to prevent broadcasting warning
        if returns_tensor.dim() == 0 or returns_tensor.size(0) == 1:
            # For single samples, use direct calculation
            critic_loss = (values - returns_tensor).pow(2).mean()
        else:
            # For batches, use MSE loss
            critic_loss = F.mse_loss(values, returns_tensor)
            
        entropy_loss = -entropies.mean()
        
        # Total loss
        loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
    
    def _compute_returns(self, rewards, dones):
        """Compute discounted returns for a batch of episodes"""
        returns = []
        
        # Get value estimate of final state
        if len(dones) > 0 and not dones[-1]:
            with torch.no_grad():
                state = torch.FloatTensor(self.env.state).unsqueeze(0).to(device)
                _, _, next_value = self.network(state)
                next_value = next_value.item()
        else:
            next_value = 0.0
        
        # Compute returns
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        
        # Normalize returns more robustly for high variance rewards
        if len(returns) > 1:
            returns = np.array(returns)
            # Use a more robust normalization with clipping
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            # Clip to avoid extremely large values
            returns = np.clip(returns, -10.0, 10.0)
            returns = returns.tolist()
        
        return returns
    
    def _update_plot(self, ax1, ax2, rewards, avg_rewards, successes):
        """Update live training plot"""
        ax1.clear()
        ax1.plot(rewards, 'b-', alpha=0.3, label='Reward')
        ax1.plot(avg_rewards, 'r-', label='Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        ax2.clear()
        # Calculate success rate over window of 10 episodes
        success_rates = []
        window_size = 10
        for i in range(len(successes)):
            if i < window_size - 1:
                rate = np.mean(successes[:i+1])
            else:
                rate = np.mean(successes[i-window_size+1:i+1])
            success_rates.append(rate)
        ax2.plot(success_rates, 'g-', label='Success Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True)
        
        plt.draw()
        plt.pause(0.01)
    
    def _plot_training_results(self, rewards, avg_rewards, successes):
        """Plot final training results"""
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 1, 1)
        plt.plot(rewards, 'b-', alpha=0.3, label='Reward')
        plt.plot(avg_rewards, 'r-', label='Moving Avg (10 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True)
        
        # Plot success rate
        plt.subplot(2, 1, 2)
        # Calculate success rate over window of 10 episodes
        success_rates = []
        window_size = 10
        for i in range(len(successes)):
            if i < window_size - 1:
                rate = np.mean(successes[:i+1])
            else:
                rate = np.mean(successes[i-window_size+1:i+1])
            success_rates.append(rate)
        plt.plot(success_rates, 'g-', label='Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title('Window Passing Success Rate')
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.checkpoint_dir}/training_results.png")
        plt.show()
    
    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f"{self.checkpoint_dir}/a2c_checkpoint_{episode}.pt")
        print(f"Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def test(self, num_episodes=10, render=True):
        """Test the trained agent"""
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        
        # Run test episodes
        num_episodes = 5
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # Store trajectory for visualization
            trajectory = [self.env.state[:3].copy()]
            
            # Run episode
            while not done:
                # Get action from policy
                action, _ = self.network.sample_action(state)
                
                # print(action)
                # Take step in environment
                next_state, reward, done, _ = self.env.step(action)
                
                # Store position for trajectory
                trajectory.append(self.env.state[:3].copy())
                
                # Update state and stats
                state = next_state
                total_reward += reward
                steps += 1
                
                # Render environment
                if render:
                    self.env.render()
                    plt.pause(0.01)
            
            # Check if episode was successful
            success = (self.env.state[1] > 0 and not self.env._hit_wall())
            
            # Log episode results
            print(f"Test Episode {episode}, Reward: {total_reward:.1f}, Steps: {steps}, Success: {success}")
            
            # Store episode stats
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_success.append(success)
            
            # Plot trajectory
            if render:
                self._plot_trajectory(np.array(trajectory), episode, success)
        
        # Print summary
        print("\nTest Results:")
        print(f"Average Reward: {np.mean(episode_rewards):.1f}")
        print(f"Average Steps: {np.mean(episode_lengths):.1f}")
        print(f"Success Rate: {np.mean(episode_success):.2%}")
        
        return episode_rewards, episode_lengths, episode_success
    
    def _plot_trajectory(self, trajectory, episode_num, success):
        """Plot the 3D trajectory of the UAV"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set fixed axis limits
        ax.set_xlim([-10.0, 10.0])
        ax.set_ylim([-10.0, 10.0])
        ax.set_zlim([-10.0, 10.0])
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', label='Trajectory')
        
        # Mark start and end
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=100, label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', s=100, label='End')
        
        # Plot window
        window_width = 2.0
        window_height = 2.0
        window_x = [-window_width/2, window_width/2, window_width/2, -window_width/2, -window_width/2]
        window_z = [-window_height/2, -window_height/2, window_height/2, window_height/2, -window_height/2]
        ax.plot(window_x, [0]*5, window_z, 'r-', label='Window')
        
        # Plot the wall as a semi-transparent plane at y=0
        wall_size = 20.0  # Fixed size for visualization
        xx, zz = np.meshgrid(np.linspace(-wall_size/2, wall_size/2, 10), 
                          np.linspace(-wall_size/2, wall_size/2, 10))
        yy = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Title based on success
        status = "Successful" if success else "Failed"
        ax.set_title(f'Episode {episode_num} Trajectory ({status})')
        
        ax.legend()
        
        # Save the figure
        plt.savefig(f"{self.checkpoint_dir}/test_episode_{episode_num}_trajectory.png")
        plt.close(fig)

if __name__ == "__main__":
    # Global configuration variables
    CHECKPOINT_PATH = "./checkpoints/a2c_checkpoint_final.pt"  # Set to None to start fresh
    # CHECKPOINT_PATH = None  # Set to None to start fresh
    RUN_MODE = "test"       # Options: "train", "test", or "both"
    NUM_BATCH_ENVS = 32      # Number of environments to run in a batch (sequential)
    
    
    # Create environment
    env = UAVEnv(max_steps=500)
    env.seed(42)
    
    # Create A2C agent (with batch environments if NUM_BATCH_ENVS > 1)
    agent = A2CAgent(env, num_envs=NUM_BATCH_ENVS)
    
    # Load checkpoint if specified
    if CHECKPOINT_PATH:
        try:
            agent.load_checkpoint(CHECKPOINT_PATH)
            print(f"Successfully loaded checkpoint from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Run training if mode is 'train' or 'both'
    if RUN_MODE in ['train', 'both']:
        # Train agent (with rendering disabled)
        agent.learn(num_episodes=10000, update_freq=10, render=False, save_freq=500, log_freq=10)
    
    # Run testing (always uses single environment)
    if RUN_MODE in ['test', 'both']:
        # Test the trained agent
        print("\nTesting A2C agent...")
        agent.test(num_episodes=10, render=True)
        # Close environment after testing
        env.close()
    
    # DOnt print Training expectations and tips
