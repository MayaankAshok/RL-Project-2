import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib
import cv2

class UAVEnv:
    """
    UAV Navigation Environment for navigating through a window
    State: 3D position and 3D velocity of the UAV
    Action: 2D orientation angles and thrust magnitude
    """
    
    def __init__(self, max_steps=200):
        # Define the boundaries of the environment (only used for visualization)
        self.vis_range = 15.0  # Visualization range
        
        # Define the window position and dimensions (at origin, aligned with XZ plane)
        self.window_x = 0.0  # x-coordinate of window center (origin)
        self.window_y = 0.0  # y-coordinate of window center (origin)
        self.window_z = 0.0  # z-coordinate of window center (origin)
        self.window_width = 5.0   # Increased from 2.0 to 3.0
        self.window_height = 5.0  # Increased from 2.0 to 3.0
        
        # UAV properties
        self.max_velocity = 2.0
        self.max_acceleration = 1.0
        self.dt = 0.1  # time step duration
        
        # Maximum number of steps per episode
        self.max_steps = max_steps
        self.steps = 0
        
        # Define observation space (3D position + 3D velocity)
        # [x, y, z, vx, vy, vz]
        # Use large bounds since there are no actual constraints in space
        self.observation_low = np.array([-float('inf'), -float('inf'), -float('inf'), 
                                         -self.max_velocity, -self.max_velocity, -self.max_velocity])
        self.observation_high = np.array([float('inf'), float('inf'), float('inf'), 
                                          self.max_velocity, self.max_velocity, self.max_velocity])
        
        # Define action space (2 euler angles: theta, phi and thrust)
        # theta: [0, pi] - polar angle from positive z-axis
        # phi: [0, 2*pi] - azimuthal angle in x-y plane from positive x-axis
        # thrust: [0, 1] - normalized thrust magnitude
        self.action_low = np.array([0.0, 0.0, 0.0])
        self.action_high = np.array([np.pi, 2*np.pi, 1.0])
        
        # Initialize state
        self.state = None
        
        # Visualization
        self.fig = None
        self.ax = None
        self.writer = None
        self.out_filename = "uav_navigation.mp4"
    
    def get_observation_space_info(self):
        """Return information about the observation space"""
        return {
            'low': self.observation_low,
            'high': self.observation_high,
            'shape': self.observation_low.shape,
            'dtype': np.float32
        }
    
    def get_action_space_info(self):
        """Return information about the action space"""
        return {
            'low': self.action_low,
            'high': self.action_high,
            'shape': self.action_low.shape,
            'dtype': np.float32
        }
    
    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)
    
    def _is_terminal(self):
        pos = self.state[:3]
        
        # Check if UAV hit the wall (crossed YZ plane except through the window)
        if self._hit_wall():
            return True
        
        # Check if the UAV has passed through the window - terminate immediately after passing
        if pos[1] > 0 and self._passed_through_window():
            return True
        
        # Check if the UAV is too far from the origin
        if np.linalg.norm(pos) > 10.0:
            return True
        
        # Check if maximum steps reached
        if self.steps >= self.max_steps:
            return True
        
        return False
    
    def _hit_wall(self):
        """Check if the UAV hit the wall (excluding window area)"""
        pos = self.state[:3]
        prev_pos = self.prev_state[:3]
        
        # Check if trajectory crossed the XZ plane (y = 0)
        if (prev_pos[1] <= 0 and pos[1] > 0) or (prev_pos[1] >= 0 and pos[1] < 0):
            # Interpolate the x and z position at the wall plane
            t = -prev_pos[1] / (pos[1] - prev_pos[1])
            x_at_wall = prev_pos[0] + t * (pos[0] - prev_pos[0])
            z_at_wall = prev_pos[2] + t * (pos[2] - prev_pos[2])
            
            # Check if it's outside the window
            window_x_min = self.window_x - self.window_width/2
            window_x_max = self.window_x + self.window_width/2
            window_z_min = self.window_z - self.window_height/2
            window_z_max = self.window_z + self.window_height/2
            
            return not (window_x_min <= x_at_wall <= window_x_max and 
                        window_z_min <= z_at_wall <= window_z_max)
        
        return False
    
    def _passed_through_window(self):
        """Check if UAV passed through the window"""
        pos = self.state[:3]
        prev_pos = self.prev_state[:3]
        
        # Check if trajectory crossed the XZ plane (y = 0) from negative y to positive y
        if (prev_pos[1] <= 0 and pos[1] > 0):
            # Interpolate the x and z position at the wall plane
            t = -prev_pos[1] / (pos[1] - prev_pos[1])
            x_at_wall = prev_pos[0] + t * (pos[0] - prev_pos[0])
            z_at_wall = prev_pos[2] + t * (pos[2] - prev_pos[2])
            
            # Check if it passed through the window opening
            window_x_min = self.window_x - self.window_width/2
            window_x_max = self.window_x + self.window_width/2
            window_z_min = self.window_z - self.window_height/2
            window_z_max = self.window_z + self.window_height/2
            
            return (window_x_min <= x_at_wall <= window_x_max and 
                    window_z_min <= z_at_wall <= window_z_max)
        
        return False
        
    def _calculate_reward(self):
        """Calculate reward based on current state, previous state, and goal"""
        pos = self.state[:3]
        vel = self.state[3:]
        
        # Default reward (small negative reward to encourage faster completion)
        reward = -0.1  # Small penalty for each step to encourage efficiency
        
        # Add proximity reward to encourage moving closer to the window
        # Calculate distance to window center
        window_center = np.array([0.0, 0.0, 0.0])  # Window is at origin
        distance_to_window = np.linalg.norm(pos - window_center)
        
        # Smoother, more gradual reward for proximity
        proximity_reward = 0.05 * (1.0 / (1.0 + distance_to_window))
        reward += proximity_reward

        if pos[2] < -1:
            reward -=0.1
        
        
        if (np.dot(vel,-pos) > 0):
            # Add reward for moving toward the window (velocity in y-direction)
            reward += 0.05
        else:
            # Add penalty for moving away from the window (velocity in y-direction)
            reward -= 0.05



        # Add reward for moving toward the window (velocity in y-direction)
        if pos[1] < 0:  # Only before passing through
            reward += 0.2 * vel[1]  # Reward for velocity toward the window
        
        # Moderate reward for passing through the window (not too large)
        if self._passed_through_window():
            reward += 20.0  # Reduced from 100 for more stable learning
        
        # Penalty for hitting the wall
        if self._hit_wall():
            # reward = -10.0  # Reduced from -50 for more stable learning
            reward = 0  # Reduced from -50 for more stable learning
        
        # Penalty for being too far from the origin
        if np.linalg.norm(pos) > 10.0 and not self._passed_through_window():
            # reward = -10.0  # Reduced from -400 for more stable learning
            reward = 0  # Reduced from -400 for more stable learning
        
        # print(reward)
        return reward
    
    def _update_state(self, action):
        """Update state based on action (2 orientation angles and thrust)"""
        # First, clip actions to valid range
        action = np.clip(action, self.action_low, self.action_high)
        
        # Save previous state
        self.prev_state = self.state.copy()
        
        # Extract current position and velocity
        pos = self.state[:3]
        vel = self.state[3:]
        
        # Extract orientation angles and thrust from action
        theta = action[0]  # polar angle (0 to pi)
        phi = action[1]    # azimuthal angle (0 to 2pi)
        thrust_magnitude = action[2] * self.max_acceleration  # normalized thrust
        
        # Convert spherical coordinates to Cartesian direction vector
        # x = r * sin(theta) * cos(phi)
        # y = r * sin(theta) * sin(phi)
        # z = r * cos(theta)
        thrust_direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Normalize direction vector (ensure unit length)
        thrust_direction = thrust_direction / np.linalg.norm(thrust_direction)
        
        # Apply thrust in the specified direction
        acceleration = thrust_direction * thrust_magnitude
        
        # Add gravity effect (in negative z direction)
        acceleration[2] -= 0.5  # simplified gravity
        # print("acceleration", acceleration)
        # Update velocity using acceleration
        new_vel = vel + acceleration * self.dt
        
        # Clip velocity to maximum
        speed = np.linalg.norm(new_vel)
        if speed > self.max_velocity:
            new_vel = new_vel * self.max_velocity / speed
        
        # Update position using velocity
        new_pos = pos + new_vel * self.dt
        
        # Update state
        self.state = np.concatenate([new_pos, new_vel])
    
    def reset(self):
        """Reset the environment to initial state"""
        # Random initial position for better exploration
        # x = np.random.uniform(-3.0, 3.0)  # Varied starting x position
        # y = np.random.uniform(-5.0, -1.0)  # Some distance from the window
        # z = np.random.uniform(-3.0, 3.0)  # Varied starting z position

        x = np.random.uniform(-5, 5)  # Varied starting x position
        y = np.random.uniform(-6, -3)  # Some distance from the window
        z = np.random.uniform(-3, 1)  # Varied starting z position


        # Initial velocity with slight bias toward window
        vx = np.random.uniform(-0, 0)
        vy = np.random.uniform(0.2, 0.3)  # Slight bias toward window
        vz = np.random.uniform(-0, 0)
        
        self.state = np.array([x, y, z, vx, vy, vz])
        self.prev_state = self.state.copy()
        self.steps = 0
        
        return self._get_obs()
    
    def sample_action(self):
        """Sample a random action from the action space"""
        return np.random.uniform(self.action_low, self.action_high)
    
    def step(self, action):
        """Take a step in the environment"""
        self.steps += 1
        
        # Update state based on action
        self._update_state(action)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if terminal state reached
        done = self._is_terminal()
        
        # Additional info
        info = {
            'position': self.state[:3],
            'velocity': self.state[3:],
            'steps': self.steps
        }
        
        return self._get_obs(), reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        mode: 'human' (default) - display using matplotlib
            'rgb_array' - write frame directly to video
        """
        # Ensure proper backend for offscreen
        if mode == 'rgb_array':
            import matplotlib
            matplotlib.use('Agg')

        # Initialize figure on first call
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            if mode == 'human':
                plt.ion()

        # Draw scene
        self.ax.clear()
        self.ax.set_xlim([-15.0, 15.0])
        self.ax.set_ylim([-10.0, 1.0])
        self.ax.set_zlim([-15.0, 15.0])
        pos = self.state[:3]
        self.ax.scatter(*pos, color='blue', s=100, label='UAV')

        # Window frame
        window_corners = [
            [-self.window_width/2, 0, -self.window_height/2],
            [ self.window_width/2, 0, -self.window_height/2],
            [ self.window_width/2, 0,  self.window_height/2],
            [-self.window_width/2, 0,  self.window_height/2],
            [-self.window_width/2, 0, -self.window_height/2]
        ]
        window_corners = np.array(window_corners)
        self.ax.plot(window_corners[:,0], window_corners[:,1], window_corners[:,2], 'r-', linewidth=2, label='Window')

        # Wall surface
        wall_size = 20.0
        x_plane = np.linspace(-wall_size/2, wall_size/2, 20)
        z_plane = np.linspace(-wall_size/2, wall_size/2, 20)
        X, Z = np.meshgrid(x_plane, z_plane)
        Y = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.3, color='gray', rstride=1, cstride=1, linewidth=0)

        # Window opening surface
        wx1, wx2 = -self.window_width/2, self.window_width/2
        wz1, wz2 = -self.window_height/2, self.window_height/2
        WX, WZ = np.meshgrid(np.linspace(wx1, wx2, 5), np.linspace(wz1, wz2, 5))
        WY = np.zeros_like(WX)
        self.ax.plot_surface(WX, WY, WZ, alpha=0.1, color='white', rstride=1, cstride=1, linewidth=0)

        # Labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('UAV Navigation Through Window')
        self.ax.legend()

        # Draw canvas
        self.fig.canvas.draw()

        if mode == 'human':
            plt.pause(0.01)

        elif mode == 'rgb_array':
            # Grab raw RGB buffer
            buf = self.fig.canvas.tostring_argb()
            h, w = self.fig.canvas.get_width_height()[::-1]
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            frame = frame[:, :, [1, 2, 3, 0]]  
            frame = frame[:, :, :3]  

            # Lazy-init writer when size known
            if self.writer is None:
            # Use MPEG-4 codec which is more likely available
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fallback codec
                self.writer = cv2.VideoWriter(self.out_filename, fourcc, 30, (w, h))
                if not self.writer.isOpened():
                    raise RuntimeError(f"Could not open VideoWriter for {self.out_filename} at {w}x{h}; check available codecs")

            # Write BGR frame
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(bgr)

    # End of updated render method

        
    def close(self):
        """Close the environment"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            
    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        if seed is not None:
            np.random.seed(seed)
        return [seed]