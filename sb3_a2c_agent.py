import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys

# Try to import required packages with helpful error messages
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("ERROR: gymnasium package not found. Please install it using:")
    print("pip install gymnasium")
    sys.exit(1)

try:
    from stable_baselines3 import A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
except ImportError:
    print("ERROR: stable-baselines3 package not found. Please install it using:")
    print("pip install stable-baselines3[extra]")
    sys.exit(1)

# Import our Gymnasium-compatible environment
try:
    from uav_gym_env import UAVGymEnv
except ImportError:
    print("ERROR: Could not import UAVGymEnv. Make sure uav_gym_env.py is in the correct location.")
    sys.exit(1)

def make_env(rank, seed=0):
    """
    Helper function to create environment
    """
    def _init():
        env = UAVGymEnv(max_steps=500)
        obs, _ = env.reset(seed=seed + rank)
        return env
    return _init

def plot_trajectory(trajectory, episode_num, success, save_dir="./sb3_checkpoints"):
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
    
    # Make sure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(f"{save_dir}/test_episode_{episode_num}_trajectory.png")
    plt.close(fig)


def test_model(model, env, num_episodes=5, render=True):
    """Test the trained model"""
    episode_rewards = []
    episode_lengths = []
    episode_success = []
    
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Store trajectory for visualization
        trajectory = [env.state[:3].copy()]
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store position for trajectory
            trajectory.append(env.state[:3].copy())
            
            # Update stats
            total_reward += reward
            steps += 1
            
            # Render environment
            if render:
                env.render()
                plt.pause(0.01)
        
        # Check if episode was successful
        success = (env.state[1] > 0 and not env._hit_wall())
        
        # Log episode results
        print(f"Test Episode {episode}, Reward: {total_reward:.1f}, Steps: {steps}, Success: {success}")
        
        # Store episode stats
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_success.append(success)
        
        # Plot trajectory
        if render:
            plot_trajectory(np.array(trajectory), episode, success)
    
    # Print summary
    print("\nTest Results:")
    print(f"Average Reward: {np.mean(episode_rewards):.1f}")
    print(f"Average Steps: {np.mean(episode_lengths):.1f}")
    print(f"Success Rate: {np.mean(episode_success):.2%}")
    
    return episode_rewards, episode_lengths, episode_success


if __name__ == "__main__":
    # Display installation instructions if this is the first run
    print("===== UAV Environment with Stable Baselines3 A2C =====")
    print("If you encounter missing module errors, install the requirements:")
    print("pip install gymnasium stable-baselines3[extra]")
    print("=====================================================\n")
    
    # Global configuration variables
    CHECKPOINT_PATH = "./sb3_checkpoints/a2c_uav"
    RUN_MODE = "test"       # Options: "train", "test", or "both"
    NUM_ENVS = 4            # Reduced number of environments to avoid memory issues
    
    # Create directories for logs and models
    os.makedirs("./sb3_checkpoints", exist_ok=True)
    os.makedirs("./sb3_logs", exist_ok=True)
    
    try:
        # Set up the environment - ONLY using DummyVecEnv to avoid multiprocessing issues
        print(f"Creating {NUM_ENVS} environments using DummyVecEnv...")
        env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])
        
        # Create a separate environment for evaluation and testing
        eval_env = Monitor(UAVGymEnv(max_steps=500))
        
        # Set up callbacks for evaluation and saving checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./sb3_checkpoints/",
            name_prefix="a2c_uav"
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./sb3_checkpoints/best_model",
            log_path="./sb3_logs/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Create or load the model
        if os.path.exists(f"{CHECKPOINT_PATH}_1000000_steps.zip") and RUN_MODE != "train":
            print(f"Loading model from {CHECKPOINT_PATH}_1000000_steps.zip")
            model = A2C.load(f"{CHECKPOINT_PATH}_1000000_steps.zip", env=env)
        else:
            # Create a new model with custom parameters
            policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[128, 128], vf=[128, 128])]
            )
            
            # Create the A2C model
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=10,
                gamma=0.99,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./sb3_logs/",
                verbose=1
            )
        
        # Run training if mode is 'train' or 'both'
        if RUN_MODE in ['train', 'both']:
            print("Training A2C model using Stable Baselines3...")
            model.learn(
                total_timesteps=1000000,  # 1 million steps
                callback=[checkpoint_callback, eval_callback],
                tb_log_name="a2c_run",
                reset_num_timesteps=True
            )
            
            # Save the final model
            model.save(f"{CHECKPOINT_PATH}_final")
        
        # Run testing
        if RUN_MODE in ['test', 'both']:
            print("\nTesting A2C model...")
            # Create a single environment for testing
            test_env = UAVGymEnv(max_steps=500, render_mode="human")
            
            # Test the model
            test_model(model, test_env, num_episodes=5, render=True)
            
            # Close environments
            env.close()
            eval_env.close()
            test_env.close()
            
    except Exception as e:
        print(f"Error running SB3 A2C agent: {e}")
        print("\nIf you're having issues with dependencies, try installing:")
        print("pip install gymnasium==0.28.1 stable-baselines3==2.1.0")
