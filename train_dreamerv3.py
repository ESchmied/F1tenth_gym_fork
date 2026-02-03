#!/usr/bin/env python3
"""
Training script for F1tenth racing with DreamerV3.

This script trains a DreamerV3 agent on the F1tenth racing simulator.
The agent learns to control a race car using sensory inputs (LiDAR scans and pose information).

Usage:
    python train_dreamerv3.py --task Spielberg
    
    # Train on a different track:
    python train_dreamerv3.py --task Monza
    
    # Use smaller model for faster training:
    python train_dreamerv3.py --task Spielberg --model_size size12m
    
    # Resume from checkpoint:
    python train_dreamerv3.py --task Spielberg --from_checkpoint /path/to/logdir/ckpt
    
    # Debug mode with fewer resources:
    python train_dreamerv3.py --task Spielberg --debug

Available tracks:
    Spielberg, Monza, Monaco, Austin, Barcelona, Silverstone, etc.
"""

import argparse
import os
import pathlib
import sys
import traceback
from functools import partial as bind

import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DreamerV3 agent with F1tenth.")
    
    # Environment arguments
    parser.add_argument("--task", type=str, default="Spielberg", help="Track/map name.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--scan_beams", type=int, default=32, help="Number of LiDAR beams (subsampled from 1080).")
    parser.add_argument("--render", action="store_true", default=False, help="Enable rendering.")
    
    # Training arguments
    parser.add_argument("--logdir", type=str, default="~/logdir/f1tenth", help="Directory for logs and checkpoints.")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total training steps.")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--eval_envs", type=int, default=0, help="Number of evaluation environments.")
    parser.add_argument("--train_ratio", type=float, default=32.0, help="Training updates per environment step.")
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
    
    print(f"\n{'='*60}")
    print("DreamerV3 Training with F1tenth")
    print(f"{'='*60}")
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
    
    # Generate timestamped logdir
    logdir = os.path.expanduser(args.logdir) + f'/{elements.timestamp()}'
    
    # Apply F1tenth-specific settings
    f1tenth_config = {
        'task': f'f1tenth_{args.task}',
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
    
    # Store args for environment creation
    env_args = {
        'task': args.task,
        'scan_beams': args.scan_beams,
        'render': args.render,
        'seed': args.seed,
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
    
    print("[INFO] Training complete!")


def make_env(config, env_args, index=0):
    """Create an F1tenth environment instance."""
    # Import the F1tenth wrapper
    script_dir = pathlib.Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from f1tenth import F1Tenth
    import embodied
    
    # Strip 'f1tenth_' prefix if present
    task = env_args['task']
    if task.startswith('f1tenth_'):
        task = task[8:]
    
    env = F1Tenth(
        task=task,
        num_agents=1,
        obs_type='features',
        scan_beams=env_args['scan_beams'],
        render_mode='human' if env_args['render'] and index == 0 else None,
    )
    
    # Apply standard wrappers
    env = wrap_env(env, config)
    
    return env


def wrap_env(env, config):
    """Apply standard wrappers to the environment."""
    import embodied
    
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_agent(config):
    """Create the DreamerV3 agent."""
    import elements
    import embodied
    from dreamerv3.agent import Agent
    
    # Import the F1tenth wrapper to get obs/act spaces
    script_dir = pathlib.Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from f1tenth import F1Tenth
    
    # Parse task name (remove 'f1tenth_' prefix if present)
    task = config.task
    if task.startswith('f1tenth_'):
        task = task[8:]
    
    # Create a temporary env to get spaces
    env = F1Tenth(task=task, num_agents=1, obs_type='features', scan_beams=32)
    env = wrap_env(env, config)
    
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    
    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)
    
    return Agent(obs_space, act_space, elements.Config(
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
