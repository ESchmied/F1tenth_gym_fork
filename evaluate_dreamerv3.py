#!/usr/bin/env python3
"""
Evaluate a trained DreamerV3 model on F1tenth (simulation or real car).

Works identically for both environments -- the only difference is the --real flag.
After each episode, generates diagnostic plots of observations, actions, and rewards.

Usage:
    # Simulation evaluation:
    python evaluate_dreamerv3.py ~/logdir/f1tenth/20260128T110207
    python evaluate_dreamerv3.py ~/logdir/f1tenth/run --render --episodes 10
    python evaluate_dreamerv3.py ~/logdir/f1tenth/run --task Spielberg --render

    # Real car evaluation:
    python evaluate_dreamerv3.py ~/logdir/f1tenth/run --real
    python evaluate_dreamerv3.py ~/logdir/f1tenth/run --real --max_speed 3.0 
    python3 evaluate_dreamerv3.py /logdir/real_20260310T120624/ckpt/20260310T225153F516088 --real --max_speed 3.0 
/home/emelies/logdir/real_20260310T120624/ckpt/20260310T225153F516088    

    # Save plots to a directory:
    python evaluate_dreamerv3.py ~/logdir/f1tenth/run --plot_dir ./eval_plots
"""

import sys
import pathlib
import argparse
import time
from collections import defaultdict

import numpy as np

import elements
import embodied
import jax
from dreamerv3.agent import Agent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(checkpoint_dir, task_override=None, model_size_override=None):
    """Load configuration from checkpoint directory and configs.yaml."""
    import ruamel.yaml as yaml

    import dreamerv3
    dreamerv3_path = pathlib.Path(dreamerv3.__file__).parent
    base_configs_path = dreamerv3_path / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(elements.Path(str(base_configs_path)).read())

    saved_config_path = pathlib.Path(checkpoint_dir).parent.parent / 'config.yaml'
    if not saved_config_path.exists():
        saved_config_path = pathlib.Path(checkpoint_dir).parent / 'config.yaml'

    if saved_config_path.exists():
        saved_config = yaml.YAML(typ='safe').load(saved_config_path.read_text())
        task = saved_config.get('task', 'f1tenth_Spielberg')
        model_size = None
        if model_size_override is None:
            for size in ['size1m', 'size12m', 'size25m', 'size50m',
                         'size100m', 'size200m', 'size400m']:
                if size in configs and 'rssm' in configs.get(size, {}):
                    if 'agent' in saved_config and 'rssm' in saved_config['agent']:
                        saved_rssm = saved_config['agent']['rssm']
                        preset_rssm = configs[size].get('agent', {}).get('rssm', {})
                        if saved_rssm.get('deter') == preset_rssm.get('deter'):
                            model_size = size
                            break
        else:
            model_size = model_size_override
    else:
        task = 'f1tenth_Spielberg'
        model_size = model_size_override

    config = elements.Config(configs['defaults'])

    if model_size and model_size in configs:
        config = config.update(configs[model_size])
        print(f"Model size: {model_size}" + (" (specified)" if model_size_override else " (detected)"))
    else:
        config = config.update(configs['size12m'])
        print("Model size: size12m (default)")

    if task_override:
        if task_override.startswith('f1tenth_'):
            task_override = task_override[8:]
        task = f'f1tenth_{task_override}'

    config = config.update(task=task)

    if saved_config_path.exists():
        if 'batch_size' in saved_config:
            config = config.update(batch_size=saved_config['batch_size'])
        if 'batch_length' in saved_config:
            config = config.update(batch_length=saved_config['batch_length'])
        if 'seed' in saved_config:
            config = config.update(seed=saved_config['seed'])

    return config


# ---------------------------------------------------------------------------
# Environment creation (sim + real, same wrapping as training)
# ---------------------------------------------------------------------------

def create_env(args, config):
    """
    Create and wrap the environment (simulation or real car).

    Uses the exact same wrapping pipeline as train_dreamerv3.py so that
    observation/action spaces are identical.
    """
    script_dir = pathlib.Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    if args.real:
        from f1tenth_real import F1TenthReal, F1TenthRealWithAutomaticReset

        real_kwargs = dict(
            scan_beams=args.scan_beams,
            max_speed=args.max_speed,
            step_frequency=args.step_frequency,
            collision_threshold=args.collision_threshold,
            scan_topic=args.scan_topic,
            odom_topic=args.odom_topic,
            drive_topic=args.drive_topic,
        )
        if args.automatic_reset:
            env = F1TenthRealWithAutomaticReset(
                pose_topic=args.pose_topic, **real_kwargs
            )
        else:
            env = F1TenthReal(**real_kwargs)
    else:
        from f1tenth import F1Tenth

        task = config.task
        if task and task.startswith('f1tenth_'):
            task = task[8:]
        if task == 'random':
            task = None

        env = F1Tenth(
            task=task,
            num_agents=1,
            obs_type='features',
            scan_beams=args.scan_beams,
            render_mode='human' if args.render else None,
        )

    # --- Apply wrappers (must match train_dreamerv3.py exactly) ---
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


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def create_agent(env, config, checkpoint_dir, jax_platform):
    """Instantiate the DreamerV3 agent and load checkpoint weights."""
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}

    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=str(checkpoint_dir.parent.parent),
        seed=config.seed,
        jax={**config.jax, 'platform': jax_platform},
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=0,
        replicas=1,
    ))

    print(f"Loading checkpoint from {checkpoint_dir} ...")
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(checkpoint_dir, keys=['agent'])
    print("Checkpoint loaded.")

    return agent


# ---------------------------------------------------------------------------
# Episode runner (collects full trajectory data)
# ---------------------------------------------------------------------------

def run_episode(env, agent, carry, render=False):
    """
    Run one episode and collect full trajectory data.

    Returns
    -------
    carry : updated policy carry state
    data  : dict of lists with keys for each obs/act field + 'reward', 'step'
    stats : dict with summary scalars (total_reward, length, etc.)
    """
    # Reset
    obs = env.step({
        'action': np.zeros(env.act_space['action'].shape, dtype=np.float32),
        'reset': True,
    })

    if render:
        env.render()

    obs_batched = {k: np.expand_dims(v, 0) for k, v in obs.items()}

    # Storage for the full trajectory
    data = defaultdict(list)
    step_idx = 0
    done = False

    while not done:
        carry, act, _ = agent.policy(carry, obs_batched, mode='eval')
        act_unbatched = {k: v[0] for k, v in act.items()}
        act_unbatched['reset'] = False

        # Record pre-step data
        data['step'].append(step_idx)
        # Raw (wrapped) action that goes to the env
        data['action_steer'].append(float(act_unbatched['action'][0]))
        data['action_speed'].append(float(act_unbatched['action'][1]))

        obs = env.step(act_unbatched)

        if render:
            env.render()

        # Record observations after step
        data['reward'].append(float(obs['reward']))
        data['scan'].append(np.array(obs.get('scan', []), dtype=np.float32))
        data['linear_vel_x'].append(float(obs.get('linear_vel_x', 0.0)))
        data['ang_vel_z'].append(float(obs.get('ang_vel_z', 0.0)))
        data['delta'].append(float(obs.get('delta', 0.0)))
        data['is_last'].append(bool(obs['is_last']))
        data['is_terminal'].append(bool(obs.get('is_terminal', False)))

        obs_batched = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        step_idx += 1
        done = obs['is_last']

    # Convert lists to arrays
    for k in data:
        data[k] = np.array(data[k])

    total_reward = float(data['reward'].sum())
    stats = {
        'reward': total_reward,
        'length': step_idx,
        'terminal': bool(data['is_terminal'].any()),
    }

    return carry, dict(data), stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_episode(data, episode_idx, plot_dir=None, show=True):
    """
    Generate diagnostic plots for a single episode.

    Subplots:
        1. Reward over time (+ cumulative)
        2. Actions (steering command, speed command)
        3. Observations: linear_vel_x, ang_vel_z, delta
        4. LiDAR scan heatmap over time
        5. Scan polar snapshot at start, middle, end
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = data['step']
    n = len(steps)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Episode {episode_idx + 1}  |  '
                 f'length={n}  reward={data["reward"].sum():.1f}  '
                 f'terminal={data["is_terminal"].any()}',
                 fontsize=14, fontweight='bold')

    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32)

    # ---- 1. Reward --------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, data['reward'], alpha=0.6, linewidth=0.8, label='step reward')
    ax1_twin = ax1.twinx()
    cumrew = np.cumsum(data['reward'])
    ax1_twin.plot(steps, cumrew, color='tab:orange', linewidth=1.2, label='cumulative')
    ax1.set_xlabel('step')
    ax1.set_ylabel('reward')
    ax1_twin.set_ylabel('cumulative reward')
    ax1.set_title('Reward')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # ---- 2. Actions -------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, data['action_steer'], label='steer cmd', linewidth=0.9)
    ax2.plot(steps, data['action_speed'], label='speed cmd', linewidth=0.9)
    ax2.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('step')
    ax2.set_ylabel('value (normalized)')
    ax2.set_title('Actions (after NormalizeAction wrapper)')
    ax2.legend(fontsize=8)

    # ---- 3. Observations: velocity / yaw rate / delta ---------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, data['linear_vel_x'], label='linear_vel_x', linewidth=0.9)
    ax3.plot(steps, data['ang_vel_z'], label='ang_vel_z', linewidth=0.9)
    ax3.plot(steps, data['delta'], label='delta (steer angle)', linewidth=0.9)
    ax3.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax3.set_xlabel('step')
    ax3.set_ylabel('value')
    ax3.set_title('Observations')
    ax3.legend(fontsize=8)

    # ---- 4. Action vs observation: steer cmd vs actual delta ---------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(steps, data['action_steer'], label='steer command', alpha=0.7, linewidth=0.9)
    ax4.plot(steps, data['delta'], label='actual delta', alpha=0.7, linewidth=0.9)
    ax4.set_xlabel('step')
    ax4.set_ylabel('steering angle')
    ax4.set_title('Steering: Command vs Actual')
    ax4.legend(fontsize=8)

    # ---- 5. Speed cmd vs actual vel ----------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(steps, data['action_speed'], label='speed command', alpha=0.7, linewidth=0.9)
    ax5.plot(steps, data['linear_vel_x'], label='actual vel_x', alpha=0.7, linewidth=0.9)
    ax5.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax5.set_xlabel('step')
    ax5.set_ylabel('speed (m/s)')
    ax5.set_title('Speed: Command vs Actual')
    ax5.legend(fontsize=8)

    # ---- 6. Phase portrait: vel_x vs ang_vel_z ----------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(data['linear_vel_x'], data['ang_vel_z'],
                          c=steps, cmap='viridis', s=4, alpha=0.6)
    ax6.set_xlabel('linear_vel_x (m/s)')
    ax6.set_ylabel('ang_vel_z (rad/s)')
    ax6.set_title('Phase: Velocity vs Yaw Rate')
    plt.colorbar(scatter, ax=ax6, label='step')

    # ---- 7. LiDAR scan heatmap over time -----------------------------------
    ax7 = fig.add_subplot(gs[2, 0:2])
    scans = data['scan']
    if scans.ndim == 2 and scans.shape[0] > 0:
        im = ax7.imshow(scans.T, aspect='auto', origin='lower',
                        cmap='inferno',
                        vmin=0, vmax=min(float(np.percentile(scans, 98)), 30.0))
        ax7.set_xlabel('step')
        ax7.set_ylabel('beam index')
        ax7.set_title('LiDAR Scan Heatmap (beam distance over time)')
        plt.colorbar(im, ax=ax7, label='distance (m)')
    else:
        ax7.text(0.5, 0.5, 'No scan data', transform=ax7.transAxes,
                 ha='center', va='center')

    # ---- 8. Polar scan snapshots -------------------------------------------
    ax8 = fig.add_subplot(gs[2, 2], projection='polar')
    if scans.ndim == 2 and scans.shape[0] > 0:
        num_beams = scans.shape[1]
        angles = np.linspace(-np.pi * 3 / 4, np.pi * 3 / 4, num_beams)
        snap_indices = [0, n // 2, n - 1]
        labels_snap = ['start', 'mid', 'end']
        colors_snap = ['tab:blue', 'tab:green', 'tab:red']
        for idx, lbl, col in zip(snap_indices, labels_snap, colors_snap):
            r = np.clip(scans[idx], 0, 30)
            ax8.plot(angles, r, label=f'{lbl} (t={idx})', linewidth=0.9,
                     color=col, alpha=0.7)
        ax8.set_title('Scan Snapshots (polar)', pad=15)
        ax8.legend(fontsize=7, loc='upper right')
    else:
        ax8.text(0, 0, 'No scan data', ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if plot_dir:
        plot_dir = pathlib.Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        path = plot_dir / f'episode_{episode_idx + 1:03d}.png'
        fig.savefig(path, dpi=150)
        print(f"  Plot saved: {path}")

    if show:
        plt.show(block=False)
        plt.pause(0.5)
    else:
        plt.close(fig)


def plot_summary(all_stats, plot_dir=None, show=True):
    """Bar chart + stats for all episodes."""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rewards = [s['reward'] for s in all_stats]
    lengths = [s['length'] for s in all_stats]
    n = len(rewards)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Evaluation Summary', fontsize=13, fontweight='bold')

    axes[0].bar(range(1, n + 1), rewards, color='steelblue')
    axes[0].axhline(np.mean(rewards), color='tab:red', linestyle='--',
                    label=f'mean={np.mean(rewards):.1f}')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Rewards')
    axes[0].legend()

    axes[1].bar(range(1, n + 1), lengths, color='seagreen')
    axes[1].axhline(np.mean(lengths), color='tab:red', linestyle='--',
                    label=f'mean={np.mean(lengths):.0f}')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Episode Lengths')
    axes[1].legend()

    plt.tight_layout()

    if plot_dir:
        plot_dir = pathlib.Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        path = plot_dir / 'summary.png'
        fig.savefig(path, dpi=150)
        print(f"Summary plot saved: {path}")

    if show:
        plt.show(block=False)
        plt.pause(0.5)
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate DreamerV3 F1tenth model (sim or real)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Checkpoint (required)
    parser.add_argument('checkpoint', type=str,
                        help='Path to checkpoint directory')

    # Environment mode
    parser.add_argument('--real', action='store_true', default=False,
                        help='Evaluate on the real car instead of simulation')

    # Simulation options
    parser.add_argument('--task', type=str, default=None,
                        help='Track name for simulation (e.g. Spielberg, Monza)')
    parser.add_argument('--render', action='store_true',
                        help='Render the simulation environment')

    # Shared options
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--scan_beams', type=int, default=32,
                        help='Number of LiDAR beams')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['size1m', 'size12m', 'size25m', 'size50m',
                                 'size100m', 'size200m', 'size400m'],
                        help='Model size (overrides auto-detection)')
    parser.add_argument('--jax_platform', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='JAX platform')

    # Real car options
    parser.add_argument('--max_speed', type=float, default=5.0,
                        help='Maximum speed for real car (m/s)')
    parser.add_argument('--step_frequency', type=float, default=20.0,
                        help='Step frequency for real car (Hz)')
    parser.add_argument('--collision_threshold', type=float, default=0.3,
                        help='Collision distance threshold (m)')
    parser.add_argument('--scan_topic', type=str, default='/scan',
                        help='ROS2 LiDAR scan topic')
    parser.add_argument('--odom_topic', type=str, default='/odom',
                        help='ROS2 odometry topic')
    parser.add_argument('--drive_topic', type=str, default='/drive',
                        help='ROS2 drive command topic')
    parser.add_argument('--automatic_reset', action='store_true', default=False,
                        help='Use automatic reset with external tracking')
    parser.add_argument('--pose_topic', type=str,
                        default='/vrpn_client_node/car/pose',
                        help='ROS2 external pose topic (for automatic reset)')

    # Plotting
    parser.add_argument('--plot_dir', type=str, default=None,
                        help='Directory to save plots (default: <logdir>/eval_plots)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting entirely')
    parser.add_argument('--no_show', action='store_true',
                        help='Save plots but do not display them')

    return parser.parse_args()


def main():
    args = parse_args()

    jax.config.update('jax_platform_name', args.jax_platform)

    mode_str = 'REAL CAR' if args.real else 'SIMULATION'
    print('=' * 70)
    print(f'F1tenth DreamerV3 Evaluation  [{mode_str}]')
    print('=' * 70)

    # --- Config ---
    checkpoint_dir = pathlib.Path(args.checkpoint)
    if not checkpoint_dir.exists():
        print(f'Error: checkpoint not found: {checkpoint_dir}')
        return

    config = load_config(checkpoint_dir,
                         task_override=args.task,
                         model_size_override=args.model_size)

    task_display = config.task or 'random'
    print(f'Task:        {task_display}')
    print(f'Checkpoint:  {checkpoint_dir}')
    print(f'Episodes:    {args.episodes}')
    print(f'Mode:        {mode_str}')
    print(f'Scan beams:  {args.scan_beams}')
    if args.real:
        print(f'Max speed:   {args.max_speed} m/s')
    print('=' * 70)

    # --- Env ---
    print('Creating environment ...')
    env = create_env(args, config)
    print(f'Obs space:   {list(env.obs_space.keys())}')
    print(f'Act space:   {list(env.act_space.keys())}')

    # --- Agent ---
    print('Creating agent ...')
    agent = create_agent(env, config, checkpoint_dir, args.jax_platform)
    carry = agent.init_policy(batch_size=1)

    # --- Plot directory ---
    if not args.no_plot:
        if args.plot_dir:
            plot_dir = pathlib.Path(args.plot_dir)
        else:
            plot_dir = checkpoint_dir.parent.parent / 'eval_plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f'Plots dir:   {plot_dir}')
    else:
        plot_dir = None

    show_plots = not args.no_show and not args.no_plot

    # --- Run episodes ---
    print()
    print('=' * 70)
    print('Running evaluation ...')
    print('=' * 70)

    all_stats = []
    all_data = []

    for ep in range(args.episodes):
        print(f'\nEpisode {ep + 1}/{args.episodes}')

        # Detect track name (sim only)
        track_name = _get_track_name(env)
        if track_name:
            print(f'  Track: {track_name}')

        t0 = time.time()
        carry, data, stats = run_episode(env, agent, carry, render=args.render)
        elapsed = time.time() - t0

        all_stats.append(stats)
        all_data.append(data)

        fps = stats['length'] / max(elapsed, 1e-6)
        print(f'  Reward:   {stats["reward"]:.2f}')
        print(f'  Length:   {stats["length"]}')
        print(f'  Terminal: {stats["terminal"]}')
        print(f'  Time:     {elapsed:.1f}s  ({fps:.0f} fps)')

        # Per-episode plot
        if not args.no_plot:
            plot_episode(data, ep, plot_dir=plot_dir, show=show_plots)

    # --- Summary ---
    print()
    print('=' * 70)
    print('Evaluation Summary')
    print('=' * 70)
    rewards = [s['reward'] for s in all_stats]
    lengths = [s['length'] for s in all_stats]
    print(f'Episodes:        {args.episodes}')
    print(f'Avg Reward:      {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}')
    print(f'Min / Max Reward:{np.min(rewards):.2f} / {np.max(rewards):.2f}')
    print(f'Avg Length:      {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}')
    print(f'Min / Max Length: {np.min(lengths)} / {np.max(lengths)}')
    print('=' * 70)

    if not args.no_plot:
        plot_summary(all_stats, plot_dir=plot_dir, show=show_plots)

    if show_plots:
        import matplotlib.pyplot as plt
        print('\nClose plot windows to exit (or Ctrl-C).')
        try:
            plt.show()
        except KeyboardInterrupt:
            pass

    env.close()


def _get_track_name(env):
    """Walk the wrapper chain to find the current track name (sim only)."""
    cur = env
    for _ in range(20):
        try:
            # Check directly in __dict__ to avoid triggering wrapper __getattr__
            if '_current_task' in getattr(cur, '__dict__', {}):
                return cur._current_task
        except (AttributeError, ValueError):
            pass
        
        # Try to get the wrapped environment
        try:
            cur = getattr(cur, 'env', None)
            if cur is None:
                cur = getattr(env, '_env', None)
        except (AttributeError, ValueError):
            pass
        
        if cur is None:
            break
    return None


if __name__ == '__main__':
    main()
