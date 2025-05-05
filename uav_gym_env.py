import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
from gymnasium import spaces

# Import the original UAV environment
from Env import UAVEnv

class UAVGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for the UAV environment.
    This makes it compatible with Stable Baselines3 and other RL libraries.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, max_steps=500, render_mode=None):
        super(UAVGymEnv, self).__init__()
        
        # Create the original UAV environment
        self.uav_env = UAVEnv(max_steps=max_steps)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array(self.uav_env.action_low, dtype=np.float32),
            high=np.array(self.uav_env.action_high, dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space (position + velocity = 6 dimensions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        # Set render mode
        self.render_mode = render_mode
        
        # For rendering
        self.fig = None
        self.ax = None
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and returns the initial observation.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for customizing reset behavior
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Set seed if provided
        if seed is not None:
            self.uav_env.seed(seed)
        
        # Reset the underlying environment
        observation = self.uav_env.reset()
        
        # Convert observation to numpy array
        observation = np.array(observation, dtype=np.float32)
        
        # Return observation and empty info dict (Gym API)
        return observation, {}
    
    def step(self, action):
        """
        Take a step in the environment using the given action.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation after taking the action
            reward: Reward received
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated (e.g., due to time limit)
            info: Additional information
        """
        # Take step in the original environment
        next_state, reward, done, info = self.uav_env.step(action)
        
        # Convert to numpy arrays
        next_state = np.array(next_state, dtype=np.float32)
        
        # Return with Gym API format
        return next_state, reward, done, False, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            If render_mode is 'rgb_array': RGB image of the environment
            If render_mode is 'human': None (renders to screen)
        """
        if self.render_mode is None:
            return
            
        # Call the original environment's render method
        # or implement custom rendering
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()
        
        # Get the current state
        pos = self.uav_env.state[:3]
        vel = self.uav_env.state[3:6]
        
        # Clear the axes
        self.ax.clear()
        
        # Set fixed axis limits
        self.ax.set_xlim([-10.0, 10.0])
        self.ax.set_ylim([-10.0, 10.0])
        self.ax.set_zlim([-10.0, 10.0])
        
        # Plot the UAV position
        self.ax.scatter(pos[0], pos[1], pos[2], color='blue', s=100, marker='o', label='UAV')
        
        # Plot velocity vector
        scale = 0.5  # scale for visibility
        self.ax.quiver(pos[0], pos[1], pos[2], 
                      vel[0]*scale, vel[1]*scale, vel[2]*scale, 
                      color='red', label='Velocity')
        
        # Plot window
        window_width = 5.0
        window_height = 5.0
        window_x = [-window_width/2, window_width/2, window_width/2, -window_width/2, -window_width/2]
        window_z = [-window_height/2, -window_height/2, window_height/2, window_height/2, -window_height/2]
        self.ax.plot(window_x, [0]*5, window_z, 'r-', label='Window')
        
        # Plot the wall as a semi-transparent plane at y=0
        wall_size = 20.0
        xx, zz = np.meshgrid(np.linspace(-wall_size/2, wall_size/2, 10), 
                         np.linspace(-wall_size/2, wall_size/2, 10))
        yy = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('UAV Environment')
        self.ax.legend()
        
        if self.render_mode == "human":
            # Display the figure
            plt.draw()
            plt.pause(0.001)
            return None
        elif self.render_mode == "rgb_array":
            # Render to an RGB array
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
    def close(self):
        """
        Close any resources used by the environment.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def seed(self, seed=None):
        """
        Set the seed for this environment.
        
        Args:
            seed: Seed to use
        """
        return self.uav_env.seed(seed)
    
    @property
    def state(self):
        """
        Get the current state of the environment.
        """
        return self.uav_env.state
    
    def _hit_wall(self):
        """
        Check if the UAV hit the wall.
        """
        return self.uav_env._hit_wall()


# Test the environment if this file is run directly
if __name__ == "__main__":
    env = UAVGymEnv(render_mode="human")
    obs, _ = env.reset(seed=42)
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation:", obs)
    
    # Run a random agent for 100 steps
    for _ in range(100):
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        # Print info
        print(f"Obs: {obs}, Reward: {reward:.2f}, Done: {terminated or truncated}")
        
        # Check if episode is done
        if terminated or truncated:
            print("Episode finished")
            obs, _ = env.reset()
    
    env.close()
