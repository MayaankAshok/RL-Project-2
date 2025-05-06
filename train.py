import numpy as np
import matplotlib.pyplot as plt
from Env import UAVEnv
import time
from mock_agent import run_episodes

import argparse
import os

args = argparse.ArgumentParser()
args.add_argument('--agent', type=str, default='SAC', help='Agent name')
args.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to run')
args.add_argument('--render', type=str, default="rgb_array", help='Render the environment')
args.add_argument('--reward', type=str, default="arctan", help='type of proximity reward function')
args = args.parse_args()

if args.agent == 'SAC':
    from sac import *

run_time_date = time.strftime("%Y-%m-%d_%H-%M-%S")
video_frames = []

def run_episodes(agent_class, env, num_episodes=10, render=None):
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

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            # Store position for trajectory
            trajectory.append(env.state[:3].copy())
            agent.update(256)

            total_reward += reward
            steps += 1
            state = next_state
            if render is not None:
                if render == "none":
                    pass
                else:
                    assert render in ['human', 'rgb_array'], "Render mode must be 'human' or 'rgb_array'"
                    env.render(mode=render)

        
        # Check if we passed through the window successfully
        passed_window = (env.state[1] > 0 and not env._hit_wall())
        if passed_window:
            success_count += 1
            
        print(f"Episode {i+1}: Reward={total_reward:.2f}, Steps={steps}, Success={passed_window}")

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        #! Plot trajectory
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
    
    # Ensure the f'plots/{run_time_date}' directory exists
    os.makedirs(f'plots/{run_time_date}', exist_ok=True)
    plt.savefig(f'plots/{run_time_date}/episode_{episode_num+1}_trajectory.png')
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
    # Ensure the f'plots/{run_time_date}' directory exists
    os.makedirs(f'plots/{run_time_date}', exist_ok=True)
    plt.savefig(f'plots/{run_time_date}/performance_metrics.png')
    # plt.show()  # Disabled to avoid UserWarning: FigureCanvasAgg is non-interactive

if __name__ == "__main__":
    env = UAVEnv(500,reward_type=args.reward)
    env.seed(42)
    np.random.seed(42)
    os.makedirs(f'plots/{run_time_date}', exist_ok=True)
    run_episodes(SACAgent, env, num_episodes=args.num_episodes, render=args.render)
    env.close()
