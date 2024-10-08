from gym import core, spaces
import gym
from dm_control import suite
from dm_env import specs
import dm_env
import numpy as np
import random
from dmc2gym import make


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class ContextualDMCWrapper(gym.Wrapper):
    """Wrapper for initialising DMC with a set of physics states"""
    def __init__(self, env, physics_states, env_kwargs, seed=0):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self._env_kwargs = env_kwargs
        self._before_reset_performed = False

        if isinstance(physics_states, int):
            self._initial_seed = physics_states + 1000*seed
            self._i = 0
            self._unbounded_states = True
            self._num_physics_states = 0
        else:
            self._num_physics_states = len(physics_states)
            if self._num_physics_states > 0:
                self._physics_states = physics_states
                self._i = 0

                # shuffle the order in which we encounter physics states 
                random.seed(seed)
                self._randomised_state_indices = np.arange(self._num_physics_states)
                random.shuffle(self._randomised_state_indices)
            self._unbounded_states = False

        assert isinstance(self._get_dmc_wrapper(), DMCWrapper), 'wrapped env must be a DMCWrapper'

    def _get_dmc_wrapper(self):
        _env = self.env
        while not isinstance(_env, DMCWrapper) and hasattr(_env, 'env'):
            _env = _env.env
        assert isinstance(_env, DMCWrapper), 'environment is not dmc2gym-wrapped'

        return _env
    
    def before_reset(self):
        if not self._before_reset_performed:
            if self._unbounded_states:
                current_seed = self._initial_seed + self._i
                self.env = make(**self._env_kwargs, seed=current_seed)
            else:
                if self._num_physics_states > 0:
                    # reset environment to reset timestep counters and other things
                    new_physics_seed, _ = self._physics_states[self._randomised_state_indices[self._i]]
                    self.env = make(**self._env_kwargs, seed=new_physics_seed)
            self._before_reset_performed = True

    def reset(self):
        if self._unbounded_states:
            self.before_reset()
            self._i += 1
            self._before_reset_performed = False
            return self.env.reset()
        else:
            if self._num_physics_states > 0:
                self.before_reset()
                # reset environment to reset timestep counters and other things
                _, new_physics_state = self._physics_states[self._randomised_state_indices[self._i]]
                new_physics_state = np.array(new_physics_state)
                self._i = (self._i + 1) % self._num_physics_states
                self.env.reset()

                dmc_env = self._get_dmc_wrapper()

                # change the physics engine's state
                physics_engine = dmc_env._env._physics
                physics_engine.set_state(new_physics_state)
                physics_engine.after_reset()

                # set correct Mujoco state and generate observation
                time_step = dm_env.TimeStep(
                    dm_env.StepType.FIRST, 
                    None, 
                    None, 
                    dmc_env._env._task.get_observation(dmc_env._env._physics)
                )
                dmc_env.current_state = _flatten_obs(time_step.observation)
                obs = dmc_env._get_obs(time_step)
                self._before_reset_performed = False
                return obs
            else:
                # fall back to normal DMC behaviour
                return self.env.reset()


class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        is_distracting_cs=None,
        distracting_cs_intensity=None,
        background_dataset_paths=None,
        environment_kwargs=None,
        setting_kwargs=None,
        channels_first=True
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._domain_name = domain_name
        self._task_name = task_name
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._is_distracting_cs = is_distracting_cs
        self._distracting_cs_intensity = distracting_cs_intensity
        self._background_dataset_paths = background_dataset_paths
        self._channels_first = channels_first

        # create task
        if is_distracting_cs:
            from cdmc.env.distracting_control import suite as dc_suite
            self._env = dc_suite.load(
                domain_name,
                task_name,
                task_kwargs=task_kwargs,
                visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs,
                difficulty=distracting_cs_intensity,
                dynamic=True,
                background_dataset_paths=background_dataset_paths
            )
        else:
            from dm_control import suite as dm_suite
            self._env = dm_suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs,
                setting_kwargs=setting_kwargs
            )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )
        
        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action), f'received invalid action "{action}" (in norm space)'
        action = self._convert_action(action)
        assert self._true_action_space.contains(action), f'received invalid action "{action}" (in true space)'
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
