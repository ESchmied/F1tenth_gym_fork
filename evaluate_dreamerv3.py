#!/usr/bin/env python3
"""
Evaluate a trained DreamerV3 model on F1tenth with rendering support.

This script manually loads the trained agent and runs inference in a custom loop,
allowing for rendering and detailed episode inspection.

Usage:
    python evaluate_dreamerv3.py ~/logdir/f1tenth/20260128T110207
    python evaluate_dreamerv3.py ~/logdir/f1tenth/20260128T110207 --render --episodes 10
    python evaluate_dreamerv3.py ~/logdir/f1tenth/20260128T110207 --task Spielberg --render
"""

import sys
import pathlib
import argparse
import numpy as np

import elements
import embodied
import jax
from dreamerv3.agent import Agent


def load_config(checkpoint_dir, task_override=None, model_size_override=None):
    """Load configuration from checkpoint directory and configs.yaml."""
    import ruamel.yaml as yaml
    
    # Load base configs from dreamerv3 package
    import dreamerv3
    dreamerv3_path = pathlib.Path(dreamerv3.__file__).parent
    base_configs_path = dreamerv3_path / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(elements.Path(str(base_configs_path)).read())
    
    # Load the saved config to get the task
    saved_config_path = pathlib.Path(checkpoint_dir).parent.parent / 'config.yaml'
    if not saved_config_path.exists():
        saved_config_path = pathlib.Path(checkpoint_dir).parent / 'config.yaml'
    
    if saved_config_path.exists():
        saved_config = yaml.YAML(typ='safe').load(saved_config_path.read_text())
        task = saved_config.get('task', 'f1tenth_Spielberg')
        # Also try to load model size from saved config
        model_size = None
        if model_size_override is None:
            for size in ['size1m', 'size12m', 'size25m', 'size50m', 'size100m', 'size200m', 'size400m']:
                # Check if model size config was applied (check a unique field from that preset)
                if size in configs and 'rssm' in configs[size]:
                    # Try to match the saved config's rssm settings
                    if 'agent' in saved_config and 'rssm' in saved_config['agent']:
                        saved_rssm = saved_config['agent']['rssm']
                        preset_rssm = configs[size]['agent']['rssm']
                        if saved_rssm.get('deter') == preset_rssm.get('deter'):
                            model_size = size
                            break
        else:
            model_size = model_size_override
    else:
        task = 'f1tenth_Spielberg'
        model_size = model_size_override
    
    # Build config the same way as training
    config = elements.Config(configs['defaults'])
    
    # Apply model size preset
    if model_size and model_size in configs:
        config = config.update(configs[model_size])
        if model_size_override:
            print(f"Using specified model size: {model_size}")
        else:
            print(f"Detected model size: {model_size}")
    else:
        # Default to size12m if not detected
        config = config.update(configs['size12m'])
        print(f"Using default model size: size12m")
    
    # Override task if specified
    if task_override:
        # Remove f1tenth_ prefix if present
        if task_override.startswith('f1tenth_'):
            task_override = task_override[8:]
        task = f'f1tenth_{task_override}'
    
    config = config.update(task=task)
    
    # Update with saved config settings if available
    if saved_config_path.exists():
        # Only copy essential fields, avoid overwriting structure
        if 'batch_size' in saved_config:
            config = config.update(batch_size=saved_config['batch_size'])
        if 'batch_length' in saved_config:
            config = config.update(batch_length=saved_config['batch_length'])
        if 'seed' in saved_config:
            config = config.update(seed=saved_config['seed'])
    
    return config


def create_env(task, render=False, scan_beams=32):
    """Create and wrap the F1tenth environment."""
    # Import local F1Tenth wrapper
    script_dir = pathlib.Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from f1tenth import F1Tenth
    
    # Strip f1tenth_ prefix if present
    if task.startswith('f1tenth_'):
        task = task[8:]
    
    # Create environment
    env = F1Tenth(
        task=task,
        num_agents=1,
        obs_type='features',
        scan_beams=scan_beams,
        render_mode='human' if render else None,
    )
    
    # Apply wrappers (same as in train_dreamerv3.py)
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    
    return env


def run_episode(env, agent, carry, render=False):
    """Run a single episode and return statistics."""
    # Reset environment - step with reset=True
    obs = env.step({'action': np.zeros(env.act_space['action'].shape, dtype=np.float32), 'reset': True})
    
    # Initial render after reset
    if render:
        env.render()
    
    # Add batch dimension to observations
    obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
    
    episode_reward = 0
    episode_length = 0
    done = False
    
    while not done:
        # Get action from agent (returns batched actions)
        carry, act, _ = agent.policy(carry, obs, mode='eval')
        
        # Remove batch dimension from actions
        act = {k: v[0] for k, v in act.items()}
        
        # Add reset=False (agent doesn't output this, but wrappers expect it)
        act['reset'] = False
        
        # Step environment
        obs = env.step(act)
        
        # Add batch dimension to observations
        obs = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        
        # Render if requested
        if render:
            env.render()
        
        # Track statistics (remove batch dimension for stats)
        episode_reward += obs['reward'][0]
        episode_length += 1
        done = obs['is_last'][0]
    
    return {
        'reward': float(episode_reward),
        'length': int(episode_length),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate DreamerV3 F1tenth model')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint directory')
    parser.add_argument('--task', type=str, default=None, help='Task/track name (overrides config, e.g., Monza)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--scan_beams', type=int, default=32, help='Number of LiDAR beams (subsampled from 1080)')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['size1m', 'size12m', 'size25m', 'size50m', 'size100m', 'size200m', 'size400m'],
                        help='Model size (overrides auto-detection)')
    parser.add_argument('--jax_platform', type=str, default='cpu', choices=['cpu', 'cuda'], help='JAX platform')
    args = parser.parse_args()
    
    # Set JAX platform
    jax.config.update('jax_platform_name', args.jax_platform)
    
    print("=" * 80)
    print("F1tenth DreamerV3 Evaluation (Custom Inference)")
    print("=" * 80)
    
    # Load configuration
    checkpoint_dir = pathlib.Path(args.checkpoint)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    print(f"Loading configuration...")
    config = load_config(checkpoint_dir, task_override=args.task, model_size_override=args.model_size)
    
    print(f"Task: {config.task}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {args.render}")
    print(f"Scan beams: {args.scan_beams}")
    print("=" * 80)
    
    # Create environment
    print("Creating environment...")
    env = create_env(config.task, render=args.render, scan_beams=args.scan_beams)
    print(f"Observation space keys: {list(env.obs_space.keys())}")
    print(f"Action space keys: {list(env.act_space.keys())}")
    
    # Create agent
    print("Creating agent...")
    # Filter out metadata keys from obs space and 'reset' from action space
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    
    # Create agent config
    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=str(checkpoint_dir.parent.parent),
        seed=config.seed,
        jax={**config.jax, 'platform': args.jax_platform},
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=0,
        replicas=1,
    ))
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_dir}...")
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(checkpoint_dir, keys=['agent'])
    print("Checkpoint loaded successfully!")
    
    # Initialize policy carry
    carry = agent.init_policy(batch_size=1)
    
    # Run evaluation episodes
    print("\n" + "=" * 80)
    print("Running evaluation episodes...")
    print("=" * 80 + "\n")
    
    all_rewards = []
    all_lengths = []
    
    for episode_idx in range(args.episodes):
        print(f"Episode {episode_idx + 1}/{args.episodes}...")
        
        # Run episode
        stats = run_episode(env, agent, carry, render=args.render)
        
        all_rewards.append(stats['reward'])
        all_lengths.append(stats['length'])
        
        print(f"  Reward: {stats['reward']:.2f}")
        print(f"  Length: {stats['length']}")
        print()
    
    # Print summary statistics
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Min Reward: {np.min(all_rewards):.2f}")
    print(f"Max Reward: {np.max(all_rewards):.2f}")
    print(f"Average Length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"Min Length: {np.min(all_lengths)}")
    print(f"Max Length: {np.max(all_lengths)}")
    print("=" * 80)
    
    # Close environment
    env.close()


if __name__ == '__main__':
    main()
