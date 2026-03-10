import functools
import numpy as np
import elements
import embodied


# Default list of F1tenth tracks for random selection (from maps directory)
# Note: Mexico City excluded due to naming inconsistency (dir has space, files don't)
DEFAULT_TRACKS = [
    'Austin', 'BrandsHatch', 'Budapest', 'Catalunya', 
    'Hockenheim', 'IMS', 'Melbourne', 'Montreal', 
    'Monza', 'MoscowRaceway', 'Nuerburgring', 'Oschersleben', 
    'Sakhir', 'SaoPaulo', 'Sepang', 'Shanghai', 
    'Silverstone', 'Sochi', 'Spa', 'Spielberg', 
    'YasMarina', 'Zandvoort'
]

# Observation features that are available on both simulation and real car
# All environments use these features for sim-to-real compatibility
TRANSFERABLE_FEATURES = ['scan', 'linear_vel_x', 'ang_vel_z', 'delta']


class F1Tenth(embodied.Env):
    """
    Wrapper for F1tenth gym environment to work with DreamerV3.
    
    Automatically uses transferable observations (scan, velocity, steering) that work
    on both simulation and real car for seamless sim-to-real transfer.
    
    Features:
    - Random starting positions ANYWHERE on the raceline for maximum exploration
    - 2000 step time limit per episode (added via TimeLimit wrapper)
    - Transferable observations for sim-to-real compatibility
    
    Handles:
    - Converting gymnasium's new API (reset returns obs, info) to DreamerV3's expected format
    - Converting multi-agent observations to single-agent for ego vehicle
    - Flattening observation space for DreamerV3
    - Subsampling LiDAR scans for efficiency
    """

    def __init__(
        self,
        task='Austin',
        num_agents=1,
        obs_type='features',
        obs_features=None,
        timestep=0.01,
        integrator='rk4',
        control_input=None,
        render_mode=None,
        scan_beams=12,
        resets_per_map=20,
        **kwargs
    ):
        """
        Args:
            task: Map name (e.g., 'Spielberg', 'Monza'), list of maps for random selection,
                  or None to use all available maps
            num_agents: Number of agents (1 for single agent training)
            obs_type: Type of observation ('original', 'features', 'kinematic_state', 'dynamic_state')
            obs_features: List of features for 'features' observation type
            timestep: Physics timestep
            integrator: Integration method ('rk4', 'euler')
            control_input: Control input type, list like ['speed', 'steering_angle']
            render_mode: Rendering mode for visualization
            scan_beams: Number of LiDAR beams to use (will subsample from 1080)
            resets_per_map: Number of episode resets before switching to a new random map (default: 20)
            
        Note: All environments use transferable observations (scan, linear_vel_x, ang_vel_z, delta)
              for seamless sim-to-real transfer.
        """
        # Import here to avoid dependency issues
        import gymnasium as gym
        
        # Always use transferable mode for sim-to-real compatibility
        self._transferable = True
        
        # Default observation features for features mode
        if obs_features is None and obs_type == 'features':
            # Always use features available on both sim and real car
            obs_features = TRANSFERABLE_FEATURES.copy()
        
        # Default control input - don't use list literal, create it explicitly
        if control_input is None:
            control_input = ['speed', 'steering_angle']
        
        # Ensure control_input is a list, not tuple
        if isinstance(control_input, tuple):
            control_input = list(control_input)
        
        # Handle task specification (single map, list of maps, or None for all maps)
        if task is None or (isinstance(task, str) and task.lower() == 'random'):
            self._task_list = DEFAULT_TRACKS.copy()
            self._random_tracks = True
            initial_task = np.random.choice(self._task_list)
        elif isinstance(task, (list, tuple)):
            self._task_list = list(task)
            self._random_tracks = True
            initial_task = np.random.choice(self._task_list)
        else:
            self._task_list = [task]
            self._random_tracks = False
            initial_task = task
        
        self._current_task = initial_task
        
        # Track resets for map switching
        self._resets_per_map = resets_per_map
        self._reset_count = 0  # Number of resets on current map
        
        # Store config parameters for recreation
        self._num_agents = num_agents
        self._obs_type = obs_type
        self._obs_features = obs_features
        self._timestep = timestep
        self._integrator = integrator
        self._control_input = control_input
        self._render_mode = render_mode
        self._scan_beams = scan_beams
        # Note: self._transferable is already set above
        self._kwargs = kwargs
        
        # Create config for F1tenth environment
        config = {
            'map': initial_task,
            'num_agents': num_agents,
            'timestep': timestep,
            'integrator': integrator,
            'control_input': control_input,
            'observation_config': {
                'type': obs_type,
            },
            'reset_config': {
                'type': 'rl_random_random'  # Use raceline random reset - spawns anywhere on track
            }
        }
        
        # Add features to observation config if needed
        if obs_features is not None and obs_type == 'features':
            config['observation_config']['features'] = obs_features
        
        # Create the environment
        self._env = gym.make('f1tenth_gym:f1tenth-v0', config=config, render_mode=render_mode, **kwargs)
        self._done = True
        self._info = None
        
        # Track current steering angle for 'delta' observation in transferable mode
        self._current_steering = 0.0
        
        # Calculate scan subsampling indices (uniformly sample scan_beams from 1080)
        full_scan_size = 1080
        self._scan_indices = np.linspace(0, full_scan_size - 1, scan_beams, dtype=np.int32)

    @property
    def env(self):
        return self._env

    @property
    def info(self):
        return self._info
    
    def _recreate_env(self, new_task):
        """Recreate the environment with a new map."""
        import gymnasium as gym
        
        # Close old environment
        try:
            self._env.close()
        except Exception:
            pass
        
        # Create config for new task
        config = {
            'map': new_task,
            'num_agents': self._num_agents,
            'timestep': self._timestep,
            'integrator': self._integrator,
            'control_input': self._control_input,
            'observation_config': {
                'type': self._obs_type,
            },
            'reset_config': {
                'type': 'rl_random_random'  # Random starting positions anywhere on track
            }
        }
        
        # Add features if needed
        if self._obs_features is not None and self._obs_type == 'features':
            config['observation_config']['features'] = self._obs_features
        
        # Create new environment
        self._env = gym.make('f1tenth_gym:f1tenth-v0', config=config, render_mode=self._render_mode, **self._kwargs)

    @functools.cached_property
    def obs_space(self):
        """Convert gym observation space to DreamerV3 format."""
        gym_obs_space = self._env.observation_space
        
        spaces = {}
        
        # Handle different observation types
        if self._obs_type == 'original':
            # Original observation is a dict with arrays for all agents
            # For single agent, we extract ego agent data
            if self._num_agents == 1:
                # Single agent: flatten the arrays
                for key, space in gym_obs_space.spaces.items():
                    if key == 'ego_idx':
                        continue  # Skip ego_idx for single agent
                    elif key == 'scans':
                        # Scans is (num_agents, scan_size), subsample to scan_beams
                        # Use infinite bounds to avoid space checking issues
                        spaces['scan'] = elements.Space(np.float32, (self._scan_beams,), -np.inf, np.inf)
                    elif key in ['poses_x', 'poses_y', 'poses_theta', 'linear_vels_x', 'linear_vels_y', 'ang_vels_z']:
                        # These are (num_agents,), take first agent as scalar
                        new_key = key.replace('poses_', 'pose_').replace('linear_vels_', 'linear_vel_').replace('ang_vels_', 'ang_vel_')
                        spaces[new_key] = elements.Space(np.float32, (), space.low[0], space.high[0])
                    elif key in ['collisions', 'lap_times', 'lap_counts']:
                        # These are also (num_agents,)
                        new_key = key.rstrip('s')  # Remove trailing 's'
                        spaces[new_key] = elements.Space(np.float32, (), space.low[0], space.high[0])
            else:
                # Multi-agent: keep as is but convert to elements.Space
                for key, space in gym_obs_space.spaces.items():
                    if key == 'ego_idx':
                        spaces[key] = elements.Space(np.int32, (), 0, self._num_agents)
                    elif key == 'scans':
                        # Subsample scans for multi-agent too, use infinite bounds
                        spaces[key] = elements.Space(np.float32, (self._num_agents, self._scan_beams), -np.inf, np.inf)
                    else:
                        spaces[key] = self._convert(space)
        else:
            # Features observation is nested dict: {agent_id: {feature: value}}
            # For single agent, flatten to just features
            if self._num_agents == 1:
                agent_space = gym_obs_space.spaces['agent_0']
                for key, space in agent_space.spaces.items():
                    if key == 'scan':
                        # Subsample scan from full size to scan_beams, use infinite bounds
                        spaces['scan'] = elements.Space(np.float32, (self._scan_beams,), -np.inf, np.inf)
                    elif key == 'collision':
                        # Collision is a Box with shape (), but we want it as float32 scalar
                        spaces['collision'] = elements.Space(np.float32, (), 0.0, 1.0)
                    else:
                        spaces[key] = self._convert(space)
                
                # Add delta (steering angle) for transferable mode
                if self._transferable and 'delta' not in spaces:
                    # Add small epsilon for floating point tolerance
                    delta_eps = 1e-5
                    spaces['delta'] = elements.Space(np.float32, (), -0.4189 - delta_eps, 0.4189 + delta_eps)
            else:
                # Multi-agent: flatten nested structure
                for agent_id, agent_space in gym_obs_space.spaces.items():
                    for key, space in agent_space.spaces.items():
                        flat_key = f'{agent_id}/{key}'
                        if key == 'scan':
                            # Subsample scan, use infinite bounds
                            spaces[flat_key] = elements.Space(np.float32, (self._scan_beams,), -np.inf, np.inf)
                        elif key == 'collision':
                            spaces[flat_key] = elements.Space(np.float32, (), 0.0, 1.0)
                        else:
                            spaces[flat_key] = self._convert(space)
        
        # Add standard DreamerV3 observation fields (these are metadata, not observations)
        spaces['reward'] = elements.Space(np.float32)
        spaces['is_first'] = elements.Space(bool)
        spaces['is_last'] = elements.Space(bool)
        spaces['is_terminal'] = elements.Space(bool)
        
        return spaces

    @functools.cached_property
    def act_space(self):
        """Convert gym action space to DreamerV3 format."""
        gym_act_space = self._env.action_space
        
        spaces = {}
        
        # F1tenth action space is a Box with shape (num_agents, action_dim)
        # For single agent: (1, 2) for [steering_angle, speed]
        if self._num_agents == 1:
            # Extract the single agent action dimensions
            action_dim = gym_act_space.shape[1]  # Should be 2
            single_low = gym_act_space.low[0]
            single_high = gym_act_space.high[0]
            spaces['action'] = elements.Space(
                np.float32, 
                (action_dim,), 
                single_low,
                single_high
            )
        else:
            # Multi-agent: create separate actions for each agent
            action_dim = gym_act_space.shape[1]
            for i in range(self._num_agents):
                single_low = gym_act_space.low[i]
                single_high = gym_act_space.high[i]
                spaces[f'action_{i}'] = elements.Space(
                    np.float32,
                    (action_dim,),
                    single_low,
                    single_high
                )
        
        # Add reset action (required by CheckSpaces wrapper)
        spaces['reset'] = elements.Space(bool)
        
        return spaces

    def step(self, action):
        """Execute one step in the environment."""
        # Handle reset (check if this is a reset call via _done flag or lack of action)
        if action.get('reset', False) or self._done:
            self._done = False
            self._current_steering = 0.0  # Reset steering tracking
            
            # Increment reset counter and switch map if needed
            if self._random_tracks:
                self._reset_count += 1
                
                # Switch to a new random map every N resets
                if self._reset_count >= self._resets_per_map:
                    new_task = np.random.choice(self._task_list)
                    if new_task != self._current_task:
                        # Need to recreate environment with new map
                        self._current_task = new_task
                        self._recreate_env(new_task)
                    self._reset_count = 0  # Reset counter after switching maps
            
            obs, self._info = self._env.reset()
            
            # Automatically render if render mode is enabled
            if self._render_mode is not None:
                self._env.render()
            
            return self._obs(obs, 0.0, is_first=True)
        
        # Convert action from DreamerV3 format to gym format
        # Gym expects shape (num_agents, action_dim)
        if self._num_agents == 1:
            # action['action'] is shape (2,), we need (1, 2)
            gym_action = np.array([action['action']], dtype=np.float32)
            # Track steering angle for 'delta' observation
            self._current_steering = float(action['action'][0])
        else:
            # Stack multi-agent actions
            gym_action = np.array(
                [action[f'action_{i}'] for i in range(self._num_agents)], 
                dtype=np.float32
            )
        
        # Execute step
        obs, reward, terminated, truncated, self._info = self._env.step(gym_action)
        self._done = terminated or truncated
        
        # Automatically render if render mode is enabled
        if self._render_mode is not None:
            self._env.render()
        
        return self._obs(
            obs,
            reward,
            is_last=bool(self._done),
            is_terminal=bool(terminated)  # Only true termination, not truncation
        )

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        """Convert gym observation to DreamerV3 format."""
        result = {}
        
        # Convert observation based on type
        if self._obs_type == 'original':
            if self._num_agents == 1:
                # Extract single agent data
                for key, value in obs.items():
                    if key == 'ego_idx':
                        continue
                    elif key == 'scans':
                        # Subsample scan from 1080 to scan_beams
                        full_scan = np.asarray(value[0], dtype=np.float32)
                        result['scan'] = full_scan[self._scan_indices]
                    elif key in ['poses_x', 'poses_y', 'poses_theta', 'linear_vels_x', 'linear_vels_y', 'ang_vels_z']:
                        new_key = key.replace('poses_', 'pose_').replace('linear_vels_', 'linear_vel_').replace('ang_vels_', 'ang_vel_')
                        result[new_key] = np.asarray(value[0], dtype=np.float32)
                    elif key in ['collisions', 'lap_times', 'lap_counts']:
                        new_key = key.rstrip('s')
                        result[new_key] = np.asarray(value[0], dtype=np.float32)
            else:
                # Keep multi-agent structure
                for key, value in obs.items():
                    if key == 'scans':
                        # Subsample scans for all agents
                        full_scans = np.asarray(value, dtype=np.float32)
                        result[key] = full_scans[:, self._scan_indices]
                    else:
                        result[key] = np.asarray(value, dtype=np.float32 if key != 'ego_idx' else np.int32)
        else:
            # Features observation
            if self._num_agents == 1:
                # Flatten single agent features
                for key, value in obs['agent_0'].items():
                    if key == 'scan':
                        # Subsample scan
                        full_scan = np.asarray(value, dtype=np.float32)
                        result[key] = full_scan[self._scan_indices]
                    elif key == 'collision':
                        # Convert bool/int collision to float32
                        result[key] = np.float32(value)
                    else:
                        result[key] = np.asarray(value, dtype=np.float32)
                
                # Add delta (steering angle) for transferable mode
                if self._transferable and 'delta' not in result:
                    # Clip to ensure within bounds (floating point precision issues)
                    delta_clipped = np.clip(self._current_steering, -0.4189, 0.4189)
                    result['delta'] = np.float32(delta_clipped)
            else:
                # Flatten multi-agent nested structure
                for agent_id, agent_obs in obs.items():
                    for key, value in agent_obs.items():
                        flat_key = f'{agent_id}/{key}'
                        if key == 'scan':
                            # Subsample scan
                            full_scan = np.asarray(value, dtype=np.float32)
                            result[flat_key] = full_scan[self._scan_indices]
                        elif key == 'collision':
                            result[flat_key] = np.float32(value)
                        else:
                            result[flat_key] = np.asarray(value, dtype=np.float32)
        
        # Add standard fields (metadata for DreamerV3, not actual observations)
        result['reward'] = np.float32(reward)
        result['is_first'] = is_first
        result['is_last'] = is_last
        result['is_terminal'] = is_terminal
        
        return result

    def render(self):
        """Render the environment."""
        return self._env.render()

    def close(self):
        """Close the environment."""
        try:
            self._env.close()
        except Exception:
            pass

    def _convert(self, space):
        """Convert gym space to elements Space."""
        import gymnasium as gym
        
        if hasattr(space, 'n'):
            # Discrete space
            return elements.Space(np.int32, (), 0, space.n)
        elif isinstance(space, gym.spaces.Box):
            # Continuous space
            if space.shape == ():
                return elements.Space(space.dtype, (), space.low, space.high)
            return elements.Space(space.dtype, space.shape, space.low.flat[0], space.high.flat[0])
        else:
            raise NotImplementedError(f"Space type {type(space)} not supported")
