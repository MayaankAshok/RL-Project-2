import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import sys
import gymnasium as gym

# Import UAVGymEnv
from uav_gym_env import UAVGymEnv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a wrapper for our environment to make it compatible with our training loop
class EnvWrapper:
    def __init__(self, env, seed=None):
        self.env = env
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def close(self):
        self.env.close()


# Class definitions for A2C following SB3 approach
class ActorCriticPolicy(nn.Module):
    """
    Actor-critic policy network similar to SB3's implementation
    """
    def __init__(self, observation_space, action_space):
        super(ActorCriticPolicy, self).__init__()
        
        # Get dimensions from spaces
        if isinstance(observation_space, gym.spaces.Box):
            self.obs_dim = observation_space.shape[0]
        else:
            self.obs_dim = observation_space.shape[0]
        
        if isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]
            self.action_low = torch.FloatTensor(action_space.low).to(device)
            self.action_high = torch.FloatTensor(action_space.high).to(device)
            self.continuous_actions = True
        else:
            self.action_dim = action_space.n
            self.continuous_actions = False
        
        # Shared feature extractor
        self.features_extractor = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim)  # Mean of action distribution
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Log standard deviation (for continuous actions)
        if self.continuous_actions:
            # Initialize log_std parameter
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        # Initialize weights like SB3
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in [self.features_extractor, self.policy_net, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
        
        # Special initialization for the policy output layer
        if isinstance(self.policy_net[-1], nn.Linear):
            nn.init.orthogonal_(self.policy_net[-1].weight, gain=0.01)
            nn.init.constant_(self.policy_net[-1].bias, 0.0)
        
        # Special initialization for the value output layer
        if isinstance(self.value_net[-1], nn.Linear):
            nn.init.orthogonal_(self.value_net[-1].weight, gain=1.0)
            nn.init.constant_(self.value_net[-1].bias, 0.0)
    
    def forward(self, obs):
        """Forward pass through the network"""
        # Convert observation to tensor if it's not already
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(device)
        
        # Extract features
        features = self.features_extractor(obs)
        
        # Get action mean and value
        action_mean = self.policy_net(features)
        value = self.value_net(features).squeeze(-1)
        
        # For continuous actions, use the log_std parameter
        if self.continuous_actions:
            action_std = torch.exp(self.log_std)
            dist = Normal(action_mean, action_std)
        else:
            # For discrete actions (not used in this environment)
            dist = torch.distributions.Categorical(logits=action_mean)
        
        return dist, value
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions for given observations"""
        dist, values = self.forward(obs)
        
        if self.continuous_actions:
            # For continuous actions, get log probability
            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            # For discrete actions
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            
        # Return values in a shape compatible with log_prob
        return values, log_prob, entropy
    
    def predict(self, obs, deterministic=False):
        """Get actions from the policy"""
        # Run forward pass with no_grad to avoid gradient tracking
        with torch.no_grad():
            dist, value = self.forward(obs)
            
            if self.continuous_actions:
                if deterministic:
                    actions = dist.mean
                else:
                    actions = dist.sample()
                
                # Clip actions to valid range
                actions = torch.max(torch.min(actions, self.action_high), self.action_low)
                
            else:
                if deterministic:
                    actions = torch.argmax(dist.probs, dim=1)
                else:
                    actions = dist.sample()
            
            # Detach before converting to numpy to avoid the error
            actions = actions.detach().cpu().numpy()
            value = value.detach().cpu().numpy()
                
        return actions, value


class CustomA2CAgent:
    """
    Custom A2C agent following the SB3 algorithm implementation
    """
    def __init__(
        self,
        env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        normalize_advantage=True,
        checkpoint_dir="./custom_checkpoints"
    ):
        self.env = EnvWrapper(env)
        self.gamma = gamma
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        
        # Create policy network
        self.policy = ActorCriticPolicy(env.observation_space, env.action_space).to(device)
        
        # Setup optimizer
        if use_rms_prop:
            self.optimizer = optim.RMSprop(
                self.policy.parameters(),
                lr=learning_rate,
                eps=rms_prop_eps,
                alpha=0.99
            )
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Storage for experience
        self.obs = None
        self.actions = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.next_obs = None
        
        # For saving checkpoints
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def collect_rollouts(self, env, n_rollout_steps):
        """Collect rollouts for training (aligned with SB3 implementation)"""
        # Initialize storage
        rollout_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_values = []
        rollout_log_probs = []
        rollout_dones = []
        rollout_successes = []
        
        # Get initial observation
        obs = self.obs
        
        for _ in range(n_rollout_steps):
            # Store current observation
            rollout_obs.append(obs.copy())
            
            # Get action and value from policy (vectorized)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                dist, value = self.policy(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # Convert to numpy
                action = action.cpu().numpy().flatten()
                value = value.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
            
            # Store action, value, log_prob
            rollout_actions.append(action)
            rollout_values.append(value)
            rollout_log_probs.append(log_prob)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store reward and done
            rollout_rewards.append(reward)
            rollout_dones.append(done)
            
            # Update observation
            obs = next_obs
            
            # Reset environment if episode ended
            if done:
                if env.env._passed_through_window():
                    # print("Episode successful!")
                    rollout_successes.append(1.0)
                else:
                    rollout_successes.append(0.0)
                obs = env.reset()
        
        # Save final observation
        self.obs = obs
        
        # Convert lists to numpy arrays
        rollout_obs = np.array(rollout_obs)
        rollout_actions = np.array(rollout_actions)
        rollout_rewards = np.array(rollout_rewards)
        rollout_values = np.array(rollout_values)
        rollout_log_probs = np.array(rollout_log_probs)
        rollout_dones = np.array(rollout_dones)
        
        # Compute returns and advantages
        advantages, returns = self._compute_returns_and_advantages(
            rollout_rewards, rollout_values, rollout_dones, obs
        )
        
        # Return a dictionary with all the rollout data
        return {
            "obs": rollout_obs,
            "actions": rollout_actions,
            "values": rollout_values,
            "log_probs": rollout_log_probs,
            "advantages": advantages,
            "returns": returns,
            "rewards": rollout_rewards,
            "dones": rollout_dones,
            "successes": rollout_successes,
        }
    
    def _compute_returns_and_advantages(self, rewards, values, dones, last_obs):
        """Compute returns and advantages using GAE (matching SB3 implementation)"""
        # Calculate the last value estimate from the final observation
        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(device)
            _, last_values = self.policy(last_obs_tensor)
            last_values = last_values.cpu().numpy()
        
        # Initialize arrays
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        # GAE calculation - refactored to avoid dimension issues
        last_gae_lam = 0.0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[step]
                next_values = values[step + 1]
            
            # Extract single values to avoid broadcasting issues
            if np.isscalar(next_values):
                next_val = next_values
            elif hasattr(next_values, 'item'):
                next_val = next_values.item()
            else:
                next_val = float(next_values.flatten()[0])
                
            if np.isscalar(values[step]):
                curr_val = values[step]
            elif hasattr(values[step], 'item'):
                curr_val = values[step].item()
            else:
                curr_val = float(values[step].flatten()[0])
            
            # Calculate delta (TD error)
            delta = rewards[step] + self.gamma * next_val * next_non_terminal - curr_val
            
            # Update advantage using GAE formula
            last_gae_lam = delta + self.gamma * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        
        # Compute returns as advantages + values (matching SB3)
        returns = advantages + values.flatten()
        
        # Normalize advantages if needed
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train(self, rollout_data):
        """Train the policy using collected rollouts (based on SB3 implementation)"""
        # Convert data to tensors
        obs_tensor = torch.FloatTensor(rollout_data["obs"]).to(device)
        actions_tensor = torch.FloatTensor(rollout_data["actions"]).to(device)
        returns_tensor = torch.FloatTensor(rollout_data["returns"]).to(device)
        advantages_tensor = torch.FloatTensor(rollout_data["advantages"]).to(device)
        
        # Flatten tensors to avoid shape mismatch
        returns_tensor = returns_tensor.flatten()
        advantages_tensor = advantages_tensor.flatten()
        
        # Evaluate actions
        values, log_probs, entropy = self.policy.evaluate_actions(obs_tensor, actions_tensor)
        
        # Make sure values are properly flattened to match returns
        values = values.flatten()
        
        # Policy loss using advantages
        policy_loss = -(advantages_tensor * log_probs).mean()
        
        # Value loss
        value_loss = F.mse_loss(returns_tensor, values)
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": loss.item(),
        }
    
    def learn(self, total_timesteps, log_interval=100, save_freq=10000):
        """Main learning method"""
        print("Starting custom A2C training...")
        
        # Initialize observations
        self.obs = self.env.reset()
        
        # Training statistics
        episode_rewards = [0.0]
        episode_lengths = [0]
        episode_success = []
        
        # Create lists for tracking metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Training loop
        timestep = 0
        num_episodes = 0
        
        while timestep < total_timesteps:
            # Collect rollouts
            rollout_data = self.collect_rollouts(self.env, self.n_steps)
            timestep += self.n_steps
            
            # Verify that all required keys exist in rollout_data
            required_keys = ["rewards", "dones", "obs", "actions", "values", "log_probs", "advantages", "returns", "successes"]
            for key in required_keys:
                if key not in rollout_data:
                    raise KeyError(f"Missing key '{key}' in rollout_data")
            
            # Update rewards and episode length tracking
            for i in range(len(rollout_data["rewards"])):
                episode_rewards[-1] += rollout_data["rewards"][i]
                episode_lengths[-1] += 1
                
                if rollout_data["dones"][i]:
                    
                    # Start a new episode tracker
                    episode_success.append(rollout_data["successes"].pop(0))
                    episode_rewards.append(0.0)
                    episode_lengths.append(0)
                    num_episodes += 1
            
            # Train the policy
            train_stats = self.train(rollout_data)
            policy_losses.append(train_stats["policy_loss"])
            value_losses.append(train_stats["value_loss"])
            entropy_losses.append(train_stats["entropy_loss"])
            
            # Log progress
            if timestep % log_interval == 0 or timestep == total_timesteps:
                # Calculate success rate
                success_rate = np.mean(episode_success[-20:]) if episode_success else 0.0
                success_count = sum(episode_success[-20:]) if episode_success else 0
                recent_count = min(len(episode_success), 20)
                
                # Calculate rewards
                if len(episode_rewards) > 1:
                    recent_rewards = episode_rewards[-11:-1]  # Last 10 complete episodes
                else:
                    recent_rewards = [0.0]
                
                print(f"Step: {timestep}/{total_timesteps}, "
                     f"Episodes: {num_episodes}, "
                     f"Avg Reward: {np.mean(recent_rewards):.2f}, "
                     f"Success: {success_count}/{recent_count} ({success_rate:.2%})")
                
            # Save checkpoint
            if timestep % save_freq == 0 or timestep == total_timesteps:
                self.save(f"{self.checkpoint_dir}/custom_a2c_{timestep}_steps.pt")
                
                # Save training metrics
                training_data = {
                    "timesteps": timestep,
                    "episode_rewards": episode_rewards,
                    "episode_lengths": episode_lengths,
                    "episode_success": episode_success,
                    "policy_losses": policy_losses,
                    "value_losses": value_losses,
                    "entropy_losses": entropy_losses
                }
                np.save(f"{self.checkpoint_dir}/training_stats_{timestep}.npy", training_data)
        
        # Save final model
        self.save(f"{self.checkpoint_dir}/custom_a2c_final.pt")
        
        # Plot training results
        self._plot_training_results(episode_rewards[:-1], episode_success)
        
        print("Training complete!")
        return episode_rewards[:-1], episode_lengths[:-1], episode_success
    
    def save(self, path):
        """Save the model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the model with proper error handling and weights_only parameter"""
        try:
            checkpoint = torch.load(path, weights_only=True)  # Use weights_only=True to avoid security warning
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
        except RuntimeError as e:
            # Fallback to the older loading method if the above fails
            print(f"Warning: Could not load with weights_only=True. Trying without this parameter.")
            checkpoint = torch.load(path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
    
    def _plot_training_results(self, rewards, successes):
        """Plot training results"""
        plt.figure(figsize=(12, 8))
        
        # Plot rewards
        plt.subplot(2, 1, 1)
        plt.plot(rewards)
        plt.title('Episode Rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.grid(True)
        
        # Calculate and plot moving average
        window_size = min(10, len(rewards))
        if window_size > 0:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg, 'r-', label=f'Moving Avg ({window_size} episodes)')
            plt.legend()
        
        # Plot success rate if available
        if successes and any(s is not None for s in successes):
            plt.subplot(2, 1, 2)
            valid_successes = [s for s in successes if s is not None]
            
            # Calculate success rate over a moving window
            window_size = min(10, len(valid_successes))
            if window_size > 0:
                success_rates = []
                for i in range(len(valid_successes)):
                    if i < window_size:
                        rate = np.mean(valid_successes[:i+1])
                    else:
                        rate = np.mean(valid_successes[i-window_size+1:i+1])
                    success_rates.append(rate)
                
                plt.plot(success_rates)
                plt.title('Success Rate (Moving Window)')
                plt.ylabel('Success Rate')
                plt.xlabel('Episode')
                plt.ylim([0, 1])
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.checkpoint_dir}/training_results.png")
    
    def test(self, num_episodes=5, deterministic=True, render=True):
        """Test the trained agent"""
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        
        for episode in range(1, num_episodes + 1):
            # Reset environment
            obs = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # Store trajectory for visualization
            trajectory = []
            if hasattr(self.env.env, 'state'):
                trajectory.append(self.env.env.state[:3].copy())
            
            # Run episode with explicit no_grad to prevent gradient computation
            with torch.no_grad():
                while not done:
                    # Get action from policy
                    action, _ = self.policy.predict(obs, deterministic=deterministic)
                    
                    # Take step in environment
                    next_obs, reward, done, info = self.env.step(action)
                    
                    # Store position for trajectory
                    if hasattr(self.env.env, 'state'):
                        trajectory.append(self.env.env.state[:3].copy())
                    
                    # Update state and stats
                    obs = next_obs
                    total_reward += reward
                    steps += 1
                    
                    # Render environment if requested and available
                    if render:
                        self.env.env.render()
                        plt.pause(0.01)
            
            # Check if episode was successful (same criteria as training)
            success = False
            try:
                # Get the final state
                state = self.env.env.state
                
                # Simple success criteria: on positive side and didn't hit wall
                on_positive_side = state[1] > 0
                hit_wall = self.env.env._hit_wall() if hasattr(self.env.env, '_hit_wall') else False
                
                # Success is defined as reaching positive Y side without hitting wall
                success = on_positive_side and not hit_wall
                
                # Always print debug info for test episodes
                print(f"\nTest episode {episode} completion:")
                print(f"  - Final position: ({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f})")
                print(f"  - On positive side: {on_positive_side}")
                print(f"  - Hit wall: {hit_wall}")
                print(f"  - Success: {success}")
                
            except Exception as e:
                print(f"Error checking test success: {str(e)}")
                success = False  # Default to failure on error
            
            # Log episode results
            print(f"Test Episode {episode}, Reward: {total_reward:.1f}, Steps: {steps}, Success: {success}")
            
            # Store episode stats
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_success.append(success)
            
            # Plot trajectory
            if render and trajectory:
                self._plot_trajectory(np.array(trajectory), episode, success)
        
        # Print summary
        print("\nTest Results:")
        print(f"Average Reward: {np.mean(episode_rewards):.1f}")
        print(f"Average Steps: {np.mean(episode_lengths):.1f}")
        print(f"Success Rate: {np.mean(episode_success):.2%} ({sum(episode_success)}/{len(episode_success)})")
        
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
        
        # Get window dimensions from the environment if available
        window_width = 2.0
        window_height = 2.0
        
        if hasattr(self.env.env, 'window_width'):
            window_width = self.env.env.window_width
        else : print("Warning: window_width not found in environment.")
        if hasattr(self.env.env, 'window_height'):
            window_height = self.env.env.window_height
            
        # Plot window
        window_x = [-window_width/2, window_width/2, window_width/2, -window_width/2, -window_width/2]
        window_z = [-window_height/2, -window_height/2, window_height/2, window_height/2, -window_height/2]
        ax.plot(window_x, [0]*5, window_z, 'r-', label='Window')
        
        # Plot the wall as a semi-transparent plane at y=0
        wall_size = 20.0
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
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        plt.savefig(f"{self.checkpoint_dir}/test_episode_{episode_num}_trajectory.png")
        plt.close(fig)


if __name__ == "__main__":
    # Enable debug mode for more detailed error reporting
    DEBUG = True
    
    # Global configuration variables
    CHECKPOINT_PATH = "./custom_checkpoints/custom_a2c_final.pt"
    RUN_MODE = "both"       # Options: "train", "test", or "both"
    
    # Create and configure the environment
    env = UAVGymEnv(max_steps=500)
    
    # Add helper method to environment if needed
    if not hasattr(env, '_hit_wall'):
        def _hit_wall(self):
            """Check if UAV hit the wall"""
            # A simple implementation - if the UAV crossed y=0, check if it went through the window
            if self.state[1] > 0 and self.prev_state[1] <= 0:
                # Check if it passed through the window
                window_width = 2.0  # Assuming window width is 2.0
                window_height = 2.0  # Assuming window height is 2.0
                in_window_x = abs(self.state[0]) <= window_width / 2
                in_window_z = abs(self.state[2]) <= window_height / 2
                
                # If it's not in the window, it hit the wall
                return not (in_window_x and in_window_z)
            return False
        
        # Add method to environment
        setattr(env.__class__, '_hit_wall', _hit_wall)
        
        # Also store previous state for this to work
        env.prev_state = env.state.copy()
        
        # Override step method to keep track of previous state
        original_step = env.step
        
        def step_with_prev_state(self, action):
            self.prev_state = self.state.copy()
            return original_step(action)
        
        env.step = types.MethodType(step_with_prev_state, env)
    
    # Create the custom A2C agent with SB3-like hyperparameters
    agent = CustomA2CAgent(
        env=env,
        learning_rate=7e-4,  # Match SB3 default
        n_steps=5,
        gamma=0.99,
        ent_coef=0.0,  # SB3 default is 0.0
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,  # SB3 default is True
        normalize_advantage=False,  # SB3 default is False
        checkpoint_dir="./custom_checkpoints"
    )
    
    # Load checkpoint if specified and exists
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        try:
            agent.load(CHECKPOINT_PATH)
            print(f"Successfully loaded checkpoint from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Will train from scratch instead.")
    
    # Run training if mode is 'train' or 'both'
    if RUN_MODE in ['train', 'both']:
        print("\nTraining custom A2C agent...")
        agent.learn(
            total_timesteps=100000,  # 1 million steps
            log_interval=1000,
            save_freq=50000
        )
    
    # Run testing
    if RUN_MODE in ['test', 'both']:
        print("\nTesting custom A2C agent...")
        # Create new environment for testing with rendering
        test_env = UAVGymEnv(max_steps=500, render_mode="human")
        test_agent = CustomA2CAgent(
            env=test_env,
            learning_rate=3e-4,
            checkpoint_dir="./custom_checkpoints"
        )
        
        # Load the trained model
        if os.path.exists(CHECKPOINT_PATH):
            test_agent.load(CHECKPOINT_PATH)
        else:
            # Try to find the latest checkpoint
            checkpoints = [f for f in os.listdir("./custom_checkpoints") if f.endswith(".pt")]
            if checkpoints:
                latest = sorted(checkpoints)[-1]
                test_agent.load(f"./custom_checkpoints/{latest}")
        
        # Test the agent
        test_agent.test(num_episodes=5, deterministic=True, render=True)
        
        # Close environment
        test_env.close()
