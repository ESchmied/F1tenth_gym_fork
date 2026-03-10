#!/usr/bin/env python3
"""
Training script for F1tenth racing with DreamerV3.

This script trains a DreamerV3 agent on the F1tenth racing simulator or real car.
The agent learns to control a race car using sensory inputs (LiDAR scans and velocity).

All training uses transferable observations (scan, linear_vel_x, ang_vel_z, delta) 
that work on both simulation and real car for seamless sim-to-real transfer.

Usage:
    # Train in simulation on random tracks:
    python train_dreamerv3.py --model_size size12m --envs 100
    
    # Train on a specific track:
    python train_dreamerv3.py --task Spielberg --model_size size12m
    
    # Control map switching frequency (default: every 20 resets):
    python train_dreamerv3.py --resets_per_map 50
    
    # Resume from checkpoint:
    python train_dreamerv3.py --task Spielberg --from_checkpoint /path/to/logdir/ckpt
    
    # Debug mode with fewer resources:
    python train_dreamerv3.py --task Spielberg --debug
    
    # Train on real car (single environment, manual reset):
    python train_dreamerv3.py --real --max_speed 3.0 --from_checkpoint /path/to/sim_checkpoint
    
    # Finetune from simulation checkpoint on real car:
    python train_dreamerv3.py --real --from_checkpoint ~/logdir/f1tenth/sim_run/ckpt

Available tracks (simulation):
    Spielberg, Monza, Monaco, Austin, Barcelona, Silverstone, etc.
    
Map Switching:
    When training with random maps (--task None), each environment sticks with one map
    for N resets (default 20) before switching to a new random map. This helps the agent
    learn each track better. Adjust with --resets_per_map.
"""

import argparse
import os
import pathlib
import sys
import traceback
from functools import partial as bind

import numpy as np

import shutil


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DreamerV3 agent with F1tenth.")
    
    # Environment arguments
    parser.add_argument("--task", type=str, default=None, 
                        help="Track/map name (default: random selection from all tracks).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--scan_beams", type=int, default=32, help="Number of LiDAR beams (subsampled from 1080).")
    parser.add_argument("--render", action="store_true", default=False, help="Enable rendering.")
    parser.add_argument("--resets_per_map", type=int, default=20, 
                        help="Number of episode resets before switching to a new random map (default: 20).")
    
    # Real car arguments
    parser.add_argument("--real", action="store_true", default=False, 
                        help="Train on real car instead of simulation.")
    parser.add_argument("--max_speed", type=float, default=3.0, 
                        help="Maximum speed for real car (m/s). Use conservative values for safety.")
    parser.add_argument("--step_frequency", type=float, default=20.0, 
                        help="Step frequency for real car (Hz).")
    parser.add_argument("--collision_threshold", type=float, default=0.3, 
                        help="Minimum distance to consider collision (m).")
    parser.add_argument("--scan_topic", type=str, default="/scan", 
                        help="ROS2 topic for LiDAR scan.")
    parser.add_argument("--odom_topic", type=str, default="/ego_racecar/odom",  #/odom
                        help="ROS2 topic for odometry.")
    parser.add_argument("--drive_topic", type=str, default="/control",  #/drive
                        help="ROS2 topic for drive commands.")
    parser.add_argument("--automatic_reset", action="store_true", default=False,
                        help="Use automatic reset with external tracking (requires pose_topic).")
    parser.add_argument("--pose_topic", type=str, default="/vrpn_client_node/car/pose",
                        help="ROS2 topic for external pose tracking (for automatic reset).")
    
    # Training arguments
    parser.add_argument("--logdir", type=str, default="~/logdir/f1tenth", help="Directory for logs and checkpoints.")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total training steps.")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--eval_envs", type=int, default=0, help="Number of evaluation environments.")
    parser.add_argument("--train_ratio", type=float, default=8.0, help="Training updates per environment step.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--batch_length", type=int, default=64, help="Sequence length for training.")
    
    # DreamerV3 model size
    parser.add_argument("--model_size", type=str, default="size12m", 
                        choices=["size1m", "size12m", "size25m", "size50m", "size100m", "size200m", "size400m"],
                        help="Model size preset.")
    
    # Logging
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging.")
    parser.add_argument("--no_tensorboard", action="store_true", default=False, help="Disable TensorBoard logging.")
    
    # Checkpoint
    parser.add_argument("--from_checkpoint", type=str, default=None, 
                        help="Path to checkpoint dir to load weights from.")
    
    # Misc
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (single env, verbose).")
    parser.add_argument("--jax_platform", type=str, default="", choices=["cuda", "cpu", ""], help="JAX platform (empty=auto).")
    
    args, other_args = parser.parse_known_args()
    return args, other_args


def main():
    """Train DreamerV3 agent with F1tenth environment."""
    args, other_args = parse_args()
    
    # Real car mode adjustments
    if args.real:
        # Force single environment for real car
        args.envs = 1
        args.eval_envs = 0
        # Reduce train ratio for real-time training
        if args.train_ratio > 16.0:
            args.train_ratio = 16.0
    
    print(f"\n{'='*60}")
    print("DreamerV3 Training with F1tenth")
    print(f"{'='*60}")
    if args.real:
        print(f"Mode: REAL CAR")
        print(f"Max speed: {args.max_speed} m/s")
        print(f"Step frequency: {args.step_frequency} Hz")
        print(f"Topics: scan={args.scan_topic}, odom={args.odom_topic}, drive={args.drive_topic}")
    else:
        print(f"Mode: SIMULATION (transferable observations)")
        if args.task is None:
            print(f"Task: Random (all tracks)")
        else:
            print(f"Task: {args.task}")
    print(f"Num envs: {args.envs}")
    print(f"Steps: {args.steps}")
    print(f"Model size: {args.model_size}")
    if args.from_checkpoint:
        print(f"Loading weights from: {args.from_checkpoint}")
    print(f"{'='*60}\n")
    
    # Import DreamerV3 modules
    print("[INFO] Importing DreamerV3 modules...")
    import elements
    import embodied
    import ruamel.yaml as yaml
    from dreamerv3.agent import Agent
    
    # Print DreamerV3 banner
    for line in Agent.banner:
        print(line)
    
    # Load base configs from dreamerv3
    print("[INFO] Loading DreamerV3 configs...")
    import dreamerv3
    dreamerv3_path = pathlib.Path(dreamerv3.__file__).parent
    config_path = dreamerv3_path / "configs.yaml"
    
    configs = yaml.YAML(typ='safe').load(elements.Path(str(config_path)).read())
    
    # Build config from presets
    config = elements.Config(configs['defaults'])
    
    # Apply model size preset
    if args.model_size in configs:
        config = config.update(configs[args.model_size])
    
    # Apply debug preset if requested
    if args.debug:
        if 'debug' in configs:
            config = config.update(configs['debug'])
        args.envs = 1
    
    # Generate timestamped logdir with mode indicator
    mode_suffix = 'real' if args.real else 'sim'
    logdir = os.path.expanduser(args.logdir) + f'/{mode_suffix}_{elements.timestamp()}'
    
    # Apply F1tenth-specific settings
    # Store task as-is (None for random, or specific task name)
    if args.real:
        task_name = 'f1tenth_real'
    else:
        task_name = 'f1tenth_random' if args.task is None else f'f1tenth_{args.task}'
    f1tenth_config = {
        'task': task_name,
        'seed': args.seed,
        'logdir': logdir,
        'batch_size': args.batch_size,
        'batch_length': args.batch_length,
    }
    config = config.update(f1tenth_config)
    
    # Update run settings
    run_config = {
        'run.steps': args.steps,
        'run.train_ratio': args.train_ratio,
        'run.envs': args.envs,
        'run.eval_envs': args.eval_envs,
    }
    config = config.update(run_config)
     
    # Update JAX platform
    config = config.update({'jax.platform': args.jax_platform})
    
    # Configure logger outputs
    logger_outputs = ['jsonl', 'scope']
    if args.tensorboard and not args.no_tensorboard:
        logger_outputs.append('tensorboard')
    config = config.update({'logger.outputs': logger_outputs})
    
    # Apply any remaining command line arguments (DreamerV3 style --key value)
    if other_args:
        config = elements.Flags(config).parse(other_args)
    
    # Setup logging directory
    logdir_path = elements.Path(config.logdir)
    print(f'[INFO] Logdir: {logdir_path}')
    logdir_path.mkdir()
    config.save(logdir_path / 'config.yaml')

    
    # Handle checkpoint loading for transfer learning / fine-tuning
    # Copy checkpoint to new logdir so DreamerV3 can load it
    if args.from_checkpoint:
        #import shutil
        src_checkpoint = pathlib.Path(os.path.expanduser(args.from_checkpoint))
        
        # Determine the source checkpoint directory structure
        # DreamerV3 structure: logdir/ckpt/TIMESTAMP/*.pkl
        if src_checkpoint.is_dir():
            # Check if this is the ckpt/ folder or a timestamped subfolder
            if src_checkpoint.name == 'ckpt':
                # User provided the ckpt/ folder, find the latest checkpoint inside
                checkpoint_folders = sorted([d for d in src_checkpoint.iterdir() if d.is_dir()])
                if checkpoint_folders:
                    checkpoint_dir = checkpoint_folders[-1]  # Latest checkpoint
                    print(f"[INFO] Using latest checkpoint: {checkpoint_dir.name}")
                else:
                    print(f"[WARNING] No checkpoint folders found in: {src_checkpoint}")
                    checkpoint_dir = None
            else:
                # User provided a timestamped checkpoint folder directly
                checkpoint_dir = src_checkpoint
        elif src_checkpoint.is_file():
            # User provided a checkpoint file, use its parent directory
            checkpoint_dir = src_checkpoint.parent
        else:
            print(f"[WARNING] Checkpoint path not found: {src_checkpoint}")
            print("[WARNING] Starting training from scratch")
            checkpoint_dir = None
        
        if checkpoint_dir and checkpoint_dir.exists():
            dst_checkpoint_root = logdir_path / 'ckpt'
            dst_checkpoint_root.mkdir(exist_ok=True)
            
            # Use a timestamp from 1 second ago to ensure it's recognized as existing
            # DreamerV3's checkpoint manager looks for the most recent checkpoint
            import time
            time.sleep(0.1)  # Small delay to ensure timestamp uniqueness
            # Use original timestamp name to preserve checkpoint identity
            dst_checkpoint_folder = dst_checkpoint_root / checkpoint_dir.name
            
            print(f"[INFO] Copying checkpoint from: {checkpoint_dir}")
            print(f"[INFO] Checkpoint timestamp: {checkpoint_dir.name}")
            shutil.copytree(checkpoint_dir, dst_checkpoint_folder, dirs_exist_ok=True)
            
            # Count files copied
            checkpoint_files = list(dst_checkpoint_folder.glob('*'))
            num_files = len(checkpoint_files)
            print(f"[INFO] Copied {num_files} files to: {dst_checkpoint_folder}")
            print(f"[INFO]   Files: {[f.name for f in checkpoint_files]}")
            
            # Verify checkpoint files exist
            agent_pkl = pathlib.Path(str(dst_checkpoint_folder)) / 'agent.pkl'
            if agent_pkl.exists():
                file_size = agent_pkl.stat().st_size / (1024 * 1024)  # MB
                print(f"[INFO]   agent.pkl: {file_size:.2f} MB")
            
            print("[INFO]") 
            print("[INFO] *** CHECKPOINT LOADING ***")
            print(f"[INFO] Checkpoint files have been copied to: {dst_checkpoint_folder}")
            print("[INFO] DreamerV3 will load these weights when training starts.")
            print("[INFO] NOTE: You may see 'Did not find any checkpoint' - this is normal.")
            print("[INFO]       The weights will still be loaded correctly.")
            print("[INFO] ************************")
    
    # Store args for environment creation
    env_args = {
        'task': args.task,
        'scan_beams': args.scan_beams,
        'render': args.render,
        'seed': args.seed,
        'resets_per_map': args.resets_per_map,
        # Real car settings
        'real': args.real,
        'max_speed': args.max_speed,
        'step_frequency': args.step_frequency,
        'collision_threshold': args.collision_threshold,
        'scan_topic': args.scan_topic,
        'odom_topic': args.odom_topic,
        'drive_topic': args.drive_topic,
        'automatic_reset': args.automatic_reset,
        'pose_topic': args.pose_topic,
    }
    
    # Build training arguments
    print("[INFO] Building training arguments...")
    train_args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )
    
    # Run training using embodied's train function
    print("[INFO] Starting training...")
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config, env_args),
        bind(make_stream, config),
        bind(make_logger, config),
        train_args,
    )
    
    print("[INFO] Training complete! Saving odom data in logdir!")
    shutil.copy('odom_topic_log.csv' , logdir_path)


def make_env(config, env_args, index=0):
    """Create an F1tenth environment instance (simulation or real car)."""
    # Import the wrappers
    script_dir = pathlib.Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    import embodied
    
    if env_args['real']:
        # Create real car environment
        from f1tenth_real import F1TenthReal, F1TenthRealWithAutomaticReset
        
        if env_args['automatic_reset']:
            env = F1TenthRealWithAutomaticReset(
                scan_beams=env_args['scan_beams'],
                max_speed=env_args['max_speed'],
                step_frequency=env_args['step_frequency'],
                collision_threshold=env_args['collision_threshold'],
                scan_topic=env_args['scan_topic'],
                odom_topic=env_args['odom_topic'],
                drive_topic=env_args['drive_topic'],
                pose_topic=env_args['pose_topic'],
            )
        else:
            env = F1TenthReal(
                scan_beams=env_args['scan_beams'],
                max_speed=env_args['max_speed'],
                step_frequency=env_args['step_frequency'],
                collision_threshold=env_args['collision_threshold'],
                scan_topic=env_args['scan_topic'],
                odom_topic=env_args['odom_topic'],
                drive_topic=env_args['drive_topic'],
            )
    else:
        # Create simulation environment
        from f1tenth import F1Tenth
        
        # Strip 'f1tenth_' prefix if present
        task = env_args['task']
        if task and task.startswith('f1tenth_'):
            task = task[8:]
        
        # Convert 'random' to None for random track selection
        if task == 'random':
            task = None
        
        env = F1Tenth(
            task=task,
            num_agents=1,
            obs_type='features',
            scan_beams=env_args['scan_beams'],
            render_mode='human' if env_args['render'] and index == 0 else None,
            resets_per_map=env_args['resets_per_map'],
        )
    
    # Apply standard wrappers
    env = wrap_env(env, config)
    
    return env


def wrap_env(env, config):
    """Apply standard wrappers to the environment."""
    import embodied
    
    # Add TimeLimit wrapper (2000 steps per episode)
    env = embodied.wrappers.TimeLimit(env, duration=2000)
    
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_agent(config, checkpoint_path=None):
    """Create the DreamerV3 agent."""
    import elements
    import embodied
    from dreamerv3.agent import Agent
    
    # Import the wrappers to get obs/act spaces
    script_dir = pathlib.Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    
    # Always use transferable observations for sim-to-real compatibility
    # Create a lightweight mock env just for space definitions (no need to initialize full env/ROS2)
    class SpaceOnlyEnv:
        def __init__(self, scan_beams=32, max_speed=5.0, max_steering=0.4189):
            self._scan_beams = scan_beams
            self._max_speed = max_speed
            self._max_steering = max_steering
        
        @property
        def obs_space(self):
            return {
                'scan': elements.Space(np.float32, (self._scan_beams,), -np.inf, np.inf),
                'linear_vel_x': elements.Space(np.float32, (), -10.0, 30.0),
                'ang_vel_z': elements.Space(np.float32, (), -10.0, 10.0),
                'delta': elements.Space(np.float32, (), -self._max_steering, self._max_steering),
                'reward': elements.Space(np.float32),
                'is_first': elements.Space(bool),
                'is_last': elements.Space(bool),
                'is_terminal': elements.Space(bool),
            }
        
        @property
        def act_space(self):
            return {
                'action': elements.Space(
                    np.float32,
                    (2,),
                    np.array([-self._max_steering, -1.0]),
                    np.array([self._max_steering, self._max_speed])
                ),
                'reset': elements.Space(bool),
            }
        
        def close(self):
            pass
    
    env = SpaceOnlyEnv(scan_beams=32)
    
    env = wrap_env(env, config)
    
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    
    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)
    
    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))
    
    # Explicitly load checkpoint if it exists (for --from_checkpoint)
    # This ensures weights are loaded BEFORE training starts
    ckpt_dir = pathlib.Path(config.logdir) / 'ckpt'
    if ckpt_dir.exists():
        # Check if there are any checkpoint folders
        checkpoint_folders = sorted([d for d in ckpt_dir.iterdir() if d.is_dir()])
        if checkpoint_folders:
            latest_ckpt = checkpoint_folders[-1]
            # Check if this checkpoint has agent.pkl (indicates it's a valid checkpoint to load)
            if (latest_ckpt / 'agent.pkl').exists():
                print(f"\n{'='*70}")
                print(f"[CHECKPOINT] Explicitly loading checkpoint: {latest_ckpt.name}")
                print(f"{'='*70}")
                try:
                    import traceback
                    # Initialize checkpoint without path, then load with path
                    cp = elements.Checkpoint()
                    cp.agent = agent
                    cp.load(latest_ckpt, keys=['agent'])
                    print(f"[CHECKPOINT] ✓ Weights loaded successfully from: {latest_ckpt}")
                    print(f"{'='*70}\n")
                except Exception as e:
                    print(f"[WARNING] Failed to load checkpoint: {e}")
                    print(f"[WARNING] Error details:")
                    traceback.print_exc()
                    print(f"[WARNING] Training will start from scratch")
                    print(f"{'='*70}\n")
    
    return agent


def make_logger(config):
    """Create the logger."""
    import elements
    
    step = elements.Counter()
    logdir = config.logdir
    multiplier = 1
    
    outputs = []
    outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
    
    for output in config.logger.outputs:
        if output == 'jsonl':
            outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
            outputs.append(elements.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'))
        elif output == 'tensorboard':
            outputs.append(elements.logger.TensorBoardOutput(logdir, config.logger.fps))
        elif output == 'wandb':
            name = '/'.join(logdir.split('/')[-4:])
            outputs.append(elements.logger.WandBOutput(name))
        elif output == 'scope':
            outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    
    logger = elements.Logger(step, outputs, multiplier)
    return logger


def make_replay(config, folder, mode='train'):
    """Create the replay buffer."""
    import elements
    import embodied
    
    batlen = config.batch_length if mode == 'train' else config.report_length
    consec = config.consec_train if mode == 'train' else config.consec_report
    capacity = config.replay.size if mode == 'train' else config.replay.size / 10
    length = consec * batlen + config.replay_context
    assert config.batch_size * length <= capacity
    
    directory = elements.Path(config.logdir) / folder
    if config.replicas > 1:
        directory /= f'{config.replica:05}'
    kwargs = dict(
        length=length, capacity=int(capacity), online=config.replay.online,
        chunksize=config.replay.chunksize, directory=directory)
    
    if config.replay.fracs.uniform < 1 and mode == 'train':
        assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
            'Gradient scaling for low-precision training can produce invalid loss '
            'outputs that are incompatible with prioritized replay.')
        recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
        selectors = embodied.selectors
        kwargs['selector'] = selectors.Mixture(dict(
            uniform=selectors.Uniform(),
            priority=selectors.Prioritized(**config.replay.prio),
            recency=selectors.Recency(recency),
        ), config.replay.fracs)
    
    return embodied.Replay(**kwargs)


def make_stream(config, replay, mode):
    """Create the data stream from replay buffer."""
    from embodied.core import streams
    
    fn = bind(replay.sample, config.batch_size, mode)
    stream = streams.Stateless(fn)
    stream = streams.Consec(
        stream,
        length=config.batch_length if mode == 'train' else config.report_length,
        consec=config.consec_train if mode == 'train' else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == 'train'),
        contiguous=True)
    return stream


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Training failed with exception:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise
