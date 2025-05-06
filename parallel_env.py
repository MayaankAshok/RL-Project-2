import numpy as np
from Env import UAVEnv
from typing import List, Tuple, Dict, Any, Optional

class SequentialEnvs:
    """
    A wrapper that simulates multiple environments sequentially
    but presents them as if they were running in parallel.
    This avoids the complexity of multiprocessing while still 
    allowing batch operations.
    """
    def __init__(self, env_fns: List[callable], seed: Optional[int] = None):
        """
        Initialize multiple environments to run sequentially
        
        Args:
            env_fns: List of functions that create environments
            seed: Optional seed for random number generators
        """
        self.num_envs = len(env_fns)
        self.envs = [env_fn() for env_fn in env_fns]
        
        # Get action and observation space from first environment
        self.action_low = self.envs[0].action_low
        self.action_high = self.envs[0].action_high
        
        if seed is not None:
            self.seed(seed)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with the given actions
        
        Args:
            actions: Actions to take in each environment (shape: [num_envs, action_dim])
            
        Returns:
            observations: Observations from all environments
            rewards: Rewards from all environments
            dones: Done flags from all environments
            infos: Info dictionaries from all environments
        """
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, info = env.step(action)
            
            # Auto-reset environments that are done
            if done:
                obs = env.reset()
                
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return np.stack(observations), np.array(rewards), np.array(dones), infos
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        observations = [env.reset() for env in self.envs]
        return np.stack(observations)
    
    def reset_one(self, env_idx: int) -> np.ndarray:
        """Reset a specific environment by index
        
        Args:
            env_idx: Index of the environment to reset
            
        Returns:
            observation: Initial observation from the reset environment
        """
        return self.envs[env_idx].reset()
    
    def render(self, env_idx: int = 0) -> None:
        """Render a specific environment"""
        self.envs[env_idx].render()
    
    def close(self) -> None:
        """Close all environments"""
        for env in self.envs:
            env.close()
    
    def seed(self, seeds: List[int] = None) -> List[List[int]]:
        """Set seeds for all environments"""
        if seeds is None:
            seeds = [None] * self.num_envs
        elif isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)
        
        return [[s] for s in seeds]
    
    def get_states(self) -> np.ndarray:
        """Get states from all environments"""
        states = [env.state for env in self.envs]
        return np.stack(states)
    
    def get_prev_states(self) -> np.ndarray:
        """Get previous states from all environments"""
        prev_states = [env.prev_state for env in self.envs]
        return np.stack(prev_states)
    
    def check_window_pass(self) -> np.ndarray:
        """Check if UAVs passed through window in all environments"""
        results = [env._passed_through_window() for env in self.envs]
        return np.array(results)
    
    def check_wall_hit(self) -> np.ndarray:
        """Check if UAVs hit wall in all environments"""
        results = [env._hit_wall() for env in self.envs]
        return np.array(results)
    
    def sample_actions(self) -> np.ndarray:
        """Sample random actions for all environments"""
        actions = []
        for env in self.envs:
            action = env.sample_action()
            actions.append(action)
        return np.array(actions)
