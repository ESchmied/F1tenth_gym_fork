#!/usr/bin/env python3
"""
Simple example of using the F1tenth DreamerV3 wrapper directly.

This demonstrates how to interact with the wrapped environment
without running full DreamerV3 training.

The environment automatically uses transferable observations (scan, linear_vel_x,
ang_vel_z, delta) that work on both simulation and real car for sim-to-real transfer.
"""

import sys
import pathlib
import numpy as np

# Add dreamerv3 to path
folder = pathlib.Path(__file__).parent / 'dreamerv3'
sys.path.insert(0, str(folder))

from embodied.envs.f1tenth import F1Tenth


def main():
    print("Creating F1tenth environment...")
    
    # Create environment
    # Note: Environment automatically uses transferable observations
    # (scan, linear_vel_x, ang_vel_z, delta) for sim-to-real compatibility
    env = F1Tenth(
        task='Spielberg',           # Track name
        num_agents=1,               # Single agent
        obs_type='features',        # Use features observation
        timestep=0.01,              # 10ms physics timestep
        integrator='rk4',           # Runge-Kutta 4 integration
        scan_beams=12,              # Subsample LiDAR to 12 beams
    )
    
    obs_space_keys = list(env.obs_space.keys())
    # Separate actual observations from metadata
    metadata_keys = ['reward', 'is_first', 'is_last', 'is_terminal']
    obs_keys = [k for k in obs_space_keys if k not in metadata_keys]
    
    print(f"Observation space: {obs_keys}")
    print(f"Metadata fields: {metadata_keys}")
    print(f"Action space: {list(env.act_space.keys())}")
    
    # Reset environment
    print("\nResetting environment...")
    action = {
        'reset': True,
        'action': np.zeros(2, dtype=np.float32)
    }
    obs = env.step(action)
    
    print(f"Observation keys: {[k for k in obs.keys() if k not in metadata_keys]}")
    print(f"Scan shape: {obs['scan'].shape} (subsampled from 1080 to {env._scan_beams} beams)")
    print(f"Is first: {obs['is_first']}")
    
    # Run episode with random actions
    print("\nRunning episode with random actions...")
    episode_reward = 0
    step = 0
    
    while not obs['is_last'] and step < 1000:
        # Random action (speed, steering)
        action = {
            'reset': False,
            'action': np.array([
                np.random.uniform(0.5, 1.0),      # Speed: 0.5 to 1.0
                np.random.uniform(-0.3, 0.3),     # Steering: -0.3 to 0.3
            ], dtype=np.float32)
        }
        
        obs = env.step(action)
        episode_reward += obs['reward']
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}: reward={obs['reward']:.3f}, done={obs['is_last']}")
    
    print(f"\nEpisode finished after {step} steps")
    print(f"Total reward: {episode_reward:.3f}")
    
    # Close environment
    env.close()
    print("\nEnvironment closed.")


if __name__ == '__main__':
    main()
