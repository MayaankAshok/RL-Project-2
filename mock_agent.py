import numpy as np
import matplotlib.pyplot as plt
from Env import UAVEnv
import time

class RandomAgent:
    """A simple random agent that takes random actions in the UAV environment"""
    
    def __init__(self, env):
        self.env = env
        
    def select_action(self, state):
        """Select a random action"""
        return self.env.sample_action()

class HeuristicAgent:
    """A simple heuristic agent that tries to navigate through the window"""
    
    def __init__(self, env):
        self.env = env
        
    def select_action(self, state):
        """Select an action based on current state using simple heuristic
        
        The heuristic is:
        1. If we're on negative y side, try to move towards window (0,0,0)
        2. Apply upward force to counteract gravity
        3. When close to window, increase forward thrust
        """
        pos = state[:3]
        vel = state[3:]
        
        # Default action (middle of action space)
        theta = np.pi/2  # 90 degrees (horizontal)
        phi = 0.0        # along positive x-axis
        thrust = 0.8     # moderate thrust
        
        # If we're on the negative y side (before window)
        if pos[1] < 0:
            # Calculate direction to window center
            target = np.array([0.0, 0.0, 0.0])
            direction = target - pos
            
            # Normalize direction
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                
            # Convert direction to spherical coordinates
            # (approximate - a more careful conversion would be better)
            theta = np.arccos(direction[2])  # polar angle
            phi = np.arctan2(direction[1], direction[0])  # azimuthal angle
            
            # Ensure phi is in [0, 2pi]
            if phi < 0:
                phi += 2 * np.pi
            
            # Increase thrust when close to window
            dist_to_window = np.linalg.norm(pos - np.array([0, 0, 0]))
            if dist_to_window < 2.0:
                thrust = 1.0
            elif dist_to_window < 5.0:
                thrust = 0.9
        
        # Counteract gravity slightly by adjusting theta
        theta = max(0.1, theta - 0.1)
        
        return np.array([theta, phi, thrust])

def run_episodes(agent_class, env, num_episodes=10, render=True):
    """Run multiple episodes with the agent in the environment"""
    agent = agent_class(env)
    
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    # Run episodes
    for i in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Store trajectory for visualization
        trajectory = [env.state[:3].copy()]
        
        try:
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store position for trajectory
                trajectory.append(env.state[:3].copy())
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if render:
                    env.render()
                    plt.pause(0.01)  # Use plt.pause instead of time.sleep for interactive mode
                    # Handle UI events to keep the plot responsive
                    plt.draw()
        except Exception as e:
            print(f"Error during episode {i+1}: {e}")
            # Clean up figure to avoid memory leak
            plt.close('all')
            continue
        
        # Check if we passed through the window successfully
        passed_window = (env.state[1] > 0 and not env._hit_wall())
        if passed_window:
            success_count += 1
            
        print(f"Episode {i+1}: Reward={total_reward:.2f}, Steps={steps}, Success={passed_window}")
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Plot trajectory
        plot_trajectory(trajectory, i, passed_window)
    
    print(f"\nSummary:")
    print(f"Total episodes: {num_episodes}")
    success_rate = success_count/max(1, num_episodes)
    print(f"Success rate: {success_rate:.2%}")
    if episode_rewards:
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Average steps: {np.mean(episode_steps):.2f}")
    
    # Plot performance metrics
    if episode_rewards:
        plot_performance(episode_rewards, episode_steps)
    
    return episode_rewards, episode_steps

def plot_trajectory(trajectory, episode_num, success):
    """Plot the 3D trajectory of the UAV"""
    trajectory = np.array(trajectory)
    
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
    window_y = [-window_width/2, window_width/2, window_width/2, -window_width/2, -window_width/2]
    window_z = [-window_height/2, -window_height/2, window_height/2, window_height/2, -window_height/2]
    ax.plot([0, 0, 0, 0, 0], window_y, window_z, 'r-', label='Window')
    
    # Plot the wall as a semi-transparent plane at x=0
    wall_size = 20.0  # Fixed size for visualization
    y_wall = np.linspace(-wall_size/2, wall_size/2, 10)
    z_wall = np.linspace(-wall_size/2, wall_size/2, 10)
    Y_wall, Z_wall = np.meshgrid(y_wall, z_wall)
    X_wall = np.zeros_like(Y_wall)
    ax.plot_surface(X_wall, Y_wall, Z_wall, alpha=0.2, color='gray')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Title based on success
    status = "Successful" if success else "Failed"
    ax.set_title(f'Episode {episode_num+1} Trajectory ({status})')
    
    ax.legend()
    
    # Save the figure
    plt.savefig(f'episode_{episode_num+1}_trajectory.png')
    plt.close(fig)

def plot_performance(rewards, steps):
    """Plot performance metrics over episodes"""
    episodes = range(1, len(rewards) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, 'b-')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.grid(True)
    
    # Plot steps
    plt.subplot(1, 2, 2)
    plt.plot(episodes, steps, 'r-')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.show()

if __name__ == "__main__":
    # Create environment
    env = UAVEnv(max_steps=200)
    
    # Set plotting to interactive mode
    plt.ion()
    
    # Set random seed for reproducibility
    env.seed(42)
    np.random.seed(42)
    
    print("\nRunning Heuristic Agent...")
    heuristic_rewards, heuristic_steps = run_episodes(HeuristicAgent, env, num_episodes=5, render=True)
    
    # Close and reopen figures between agent runs to avoid interference
    plt.close('all')
    print("Running Random Agent...")
    random_rewards, random_steps = run_episodes(RandomAgent, env, num_episodes=5, render=True)
    
    
    # Compare performance only if both agents have completed episodes
    if random_rewards and heuristic_rewards:
        # Switch to non-interactive mode for final plots
        plt.ioff()
        
        plt.figure(figsize=(12, 5))
        
        # Plot rewards comparison
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(random_rewards) + 1), random_rewards, 'b-', label='Random')
        plt.plot(range(1, len(heuristic_rewards) + 1), heuristic_rewards, 'r-', label='Heuristic')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards: Random vs Heuristic')
        plt.legend()
        plt.grid(True)
        
        # Plot steps comparison
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(random_steps) + 1), random_steps, 'b-', label='Random')
        plt.plot(range(1, len(heuristic_rewards) + 1), heuristic_steps, 'r-', label='Heuristic')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps: Random vs Heuristic')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('agents_comparison.png')
        plt.show(block=True)  # Block execution until window is closed
    
    # Close environment
    env.close()
