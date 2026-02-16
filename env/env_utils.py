import os
os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import numpy as np
import torch
import random
import metaworld
import cv2
from typing import Optional, Tuple, Dict, Any, Literal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
from omegaconf import OmegaConf
import cv2


def _is_missing_checkpoint(x) -> bool:
    return x is None or x == "None"

def _banner(msg: str):
    print("#" * 56)
    print(msg)
    print("#" * 56)
    
def build_obs_shape_from_obs_config(obs_config):
    obs_dim = 0
    for k in obs_config.keys():
        obs_dim += np.prod(obs_config[k])
    return (obs_dim,)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _obs_dict_to_space(obs: Dict[str, np.ndarray]) -> spaces.Dict:
    """
    Build a Gymnasium Dict space that mirrors dm_control's observation dict.
    Each entry is a Box(-inf, inf, shape=v.shape, dtype=float32).
    """
    box_map = {}
    for k, v in obs.items():
        arr = np.asarray(v)
        box_map[k] = spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=np.float32)
    return spaces.Dict(box_map)

def _cast_obs_float32(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return a new dict with values cast to float32 (common for policies)."""
    return {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}


class DMCGym(gym.Env):
    """
    dm_control -> Gymnasium wrapper that preserves RAW dict observations.

    terminated/truncated semantics:
      - terminated=True  iff dm_control TimeStep.last() is True.
      - truncated=True   iff wrapper max_episode_steps reached first.

    If `infinite_time_limit=True`, dm_control's internal time limit is set to ∞,
    so episodes end only via wrapper truncation (or true task terminals, if any).
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        domain: str,
        task: str,
        *,
        n_sub_steps: int = 1,
        seed: Optional[int] = None,
        render_size: Tuple[int, int] = (240, 320),
        camera_id: int = 0,
    ):
        super().__init__()
        self.domain = domain
        self.task = task
        self._n_sub_steps = n_sub_steps
        self._step_count = 0

        self._render_h, self._render_w = render_size
        self._camera_id = camera_id

        self._last_seed = int(seed) if seed is not None else np.random.randint(0, 2**31 - 1)

        self._build_env(self._last_seed)

        # Action space
        a_spec = self.env.action_spec()
        self._a_low = a_spec.minimum.astype(np.float32)
        self._a_high = a_spec.maximum.astype(np.float32)
        self.action_space = spaces.Box(low=self._a_low, high=self._a_high, dtype=np.float32)

        # Observation space (dict)
        ts0 = self.env.reset()
        self.observation_space = _obs_dict_to_space(ts0.observation)

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _build_env(self, seed: int):
        self.env = suite.load(
            domain_name=self.domain,
            task_name=self.task,
            task_kwargs={"random": int(seed)},
            # environment_kwargs={"n_sub_steps": self._n_sub_steps}
        )
        self._max_steps = self.env._step_limit

    # ---------------------------
    # Gymnasium API
    # ---------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._last_seed = int(seed)
            self._build_env(self._last_seed)

        self._step_count = 0
        ts = self.env.reset()
        obs = _cast_obs_float32(ts.observation)
        info = {"episode_seed": self._last_seed,"ts":ts}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, self._a_low, self._a_high).astype(np.float32)

        reward = 0.0
        ts = None
        ts = self.env.step(action)
        reward += float(ts.reward or 0.0)
        self._step_count += 1
        obs = _cast_obs_float32(ts.observation)

        terminated = bool(ts.last())
        truncated = (self._step_count >= self._max_steps)
        terminated = False if truncated else terminated ### dm_control does not have terminated, so we use truncated to indicate the end of the episode
        return obs, reward, terminated, truncated, {'ts':ts}

    def render(self) -> np.ndarray:
        return self.env.physics.render(
            height=self._render_h, width=self._render_w, camera_id=self._camera_id
        )

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

class DMCWrapper(gym.Wrapper):
    def __init__(self,env:DMCGym,obs_config):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = self.unwrapped._max_steps
        self.obs_config = obs_config
        self.action_space = env.action_space
        self.env_category = 'dmc'
        
    def _parse_obs(self,obs):
        final_obs = []
        for k in self.obs_config.keys():
            final_obs.append(obs[k].reshape(-1))
        final_obs = np.concatenate(final_obs, axis=0)
        return final_obs
    
    def step(self,action):
        obs,reward,done,truncated,info = self.env.step(action)
        obs = self._parse_obs(obs)
        info['is_success'] = False 
        return obs,reward,done,truncated,info
    def reset(self,seed=None,options=None):
        o,info =self.env.reset(seed=seed,options=options)
        o = self._parse_obs(o)
        return o,info
    def close(self):
        self.unwrapped.close()
    def render(self):
        return self.unwrapped.render()
 
    
def make_dmc_env(domain,task,obs_config, seed=0,render_size = (128,128),n_sub_steps=1):
    env = DMCGym(domain,task,n_sub_steps=n_sub_steps,seed=seed,render_size=render_size)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = DMCWrapper(env,obs_config)
    return env
  
    
def make_metaworld_env(env_name,obs_config,seed=0,camera_to_render='corner4',reward_scale = 1.0, step_penality = 0.01,max_steps = 400,sparse_reward = False):
    
    env = gym.make('Meta-World/MT1', env_name=env_name, seed=seed,render_mode="rgb_array",camera_name=camera_to_render)
    env = MetaworldWrapper(env,obs_config,reward_scale,step_penality,max_steps,sparse_reward)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

class MetaworldWrapper(gym.Wrapper):
    def __init__(self,env,obs_config,reward_scale = 0.1, step_penality = 1.0,max_steps = 500,sparse_reward = False):
        super().__init__(env)
        self.step_count = 0
        self.obs_config = obs_config ## for metaworld, we only has robot states as observation robot_states: robot_states
        self.action_space = env.action_space
        self.reward_scale = reward_scale
        self.step_penality = step_penality
        self.env_category = 'metaworld'
        self.max_steps = max_steps
        self.sparse_reward = sparse_reward
            
    def _parse_obs(self,obs):
        final_obs = obs
        return final_obs
    
    def step(self,action):
        self.step_count += 1
        obs,raw_reward,done,truncated,info = self.env.step(action)
        if 'episode' in info:
            info.pop('episode')
        info['is_success'] = info['success'] > 0.0
        
        if not info['success']:
                
            if self.prev_raw_reward is not None:
                # reward = max(-0.01, (raw_reward - self.prev_raw_reward) * self.reward_scale ) - self.step_penality 
                reward = (raw_reward - self.prev_raw_reward) * self.reward_scale  - self.step_penality 
                self.prev_raw_reward = raw_reward 
            else:
                reward = 0.0 
                self.prev_raw_reward = raw_reward 
            if self.sparse_reward:
                reward = 0.0
        else:
            reward = min(raw_reward , 5) ## don't want to be too large 
        
        obs = self._parse_obs(obs)
        done = done or info['success'] > 0.0
        truncated = truncated or self.step_count >= self.max_steps
        
        return obs,reward,done,truncated,info
    def reset(self,seed=None,options=None):
        self.prev_raw_reward = None
        self.step_count = 0
        obs,info = self.env.reset()
        obs = self._parse_obs(obs)
        return obs,info
    def close(self):
        self.env.close()
    def render(self):
        image = self.env.render()
        return image[::-1, :, :]


import numpy as np
import gymnasium as gym
from collections import deque
from third_party.customized_box2d.HybridBipedalWalker import HybridBipedalWalker

class HybridWalkerWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4,max_steps=1000):
        """
        Args:
            env: The HybridBipedalWalker environment
            stack_frames: int, number of frames to stack (k)
        """
        super().__init__(env)
        self.step_count = 0
        self.stack_frames = stack_frames
        self.max_steps = max_steps
        
        # 使用 deque 作为固定长度的帧缓冲区
        self.frames = deque(maxlen=stack_frames)
        
        # --- Flatten & Stack Action/Observation Space Setup ---
        self.action_space = env.action_space
        
        # 原环境的 obs shape 是 (24,)
        # Stack 之后的 obs shape 应该是 (24 * k,)
        # 我们需要扩展 observation_space 的定义，否则 RL 算法可能会报错维度不匹配
        src_low = self.env.observation_space.low
        src_high = self.env.observation_space.high
        
        # np.tile 将原有的 limit 重复 k 次
        stacked_low = np.tile(src_low, stack_frames)
        stacked_high = np.tile(src_high, stack_frames)
        
        self.observation_space = gym.spaces.Box(
            low=stacked_low,
            high=stacked_high,
            dtype=self.env.observation_space.dtype
        )
        
        self.env_category = 'box2d_hybrid'

    def _get_flattened_obs(self):
        """
        将 deque 中的 k 个帧 (每个 24 维) 拼接并 Flatten 成一个 (24*k,) 的向量
        """
        # list(self.frames) 返回 [array(24,), array(24,), ...]
        # np.concatenate 在 axis 0 拼接，结果为 array(96,)
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        
        # 获取初始帧
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset 时，因为没有历史帧，我们用初始帧填满缓冲区
        # 这样 Learner 在 t=0 时看到的 input 是 [s0, s0, s0, s0]
        # 能够避免冷启动时的 zeros padding 带来的分布偏移
        for _ in range(self.stack_frames):
            self.frames.append(obs)
            
        final_obs = self._get_flattened_obs()
        return final_obs, info

    def step(self, action):
        self.step_count += 1
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 将新的一帧推入 buffer (最老的帧会自动弹出)
        self.frames.append(obs)
        
        # 获取 Flatten 后的 Stacked Frame
        final_obs = self._get_flattened_obs()
        if self.step_count >= self.max_steps:
            terminated = True
            truncated = True
        
        return final_obs, reward, terminated, truncated, info

    def close(self):
        self.unwrapped.close()

    def render(self):
        return self.unwrapped.render()


def make_hybrid_walker(mode="learner",stack_frames=4,max_steps=1000):
    env = HybridBipedalWalker(mode=mode,render_mode="rgb_array")
    env = HybridWalkerWrapper(env,stack_frames=stack_frames,max_steps=max_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

class HybridLanderWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4,max_steps=1000):
        """
        Args:
            env: The HybridLunarLander environment instance
            stack_frames: int, number of frames to stack (k)
        """
        super().__init__(env)
        self.stack_frames = stack_frames
        self.max_steps = max_steps
        # 使用 deque 作为一个固定长度的滑动窗口 (FIFO)
        self.frames = deque(maxlen=stack_frames)
        
        self.action_space = env.action_space
        
        # --- Observation Space Flattening ---
        # 原本 Obs Shape: (8,) -> Stack 后: (8 * k,)
        src_low = self.env.observation_space.low
        src_high = self.env.observation_space.high
        
        # 将原本的 bound 重复 k 次
        stacked_low = np.tile(src_low, stack_frames)
        stacked_high = np.tile(src_high, stack_frames)
        
        self.observation_space = gym.spaces.Box(
            low=stacked_low,
            high=stacked_high,
            dtype=self.env.observation_space.dtype
        )
        
        # 标记环境类别，方便后续做特定处理
        self.env_category = 'box2d_hybrid_lander'

    def _get_flattened_obs(self):
        """
        将 deque 中的 k 个帧拼接成一个长向量
        Return shape: (32,) if k=4
        """
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        """
        Reset 环境并用初始帧填满 Buffer
        """
        self.step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)
        
        # 冷启动：用第一帧填满整个 buffer
        # 这样 t=0 时 Learner 看到的是静止状态 [s0, s0, s0, s0]
        for _ in range(self.stack_frames):
            self.frames.append(obs)
            
        return self._get_flattened_obs(), info

    def step(self, action):
        """
        执行一步，推入新帧，并计算 is_success
        """
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 推入新的一帧，最老的一帧自动弹出
        self.frames.append(obs)
        
        # 获取 Flatten 后的 Stacked Frame
        final_obs = self._get_flattened_obs()
        
        # --- Success Metric ---
        # LunarLander 的逻辑：如果 lander.awake 为 False (休眠/静止)，
        # 且没有 crash (-100)，step reward 会被设为 +100。
        info['is_success'] = False
        if terminated and reward == 100: 
             info['is_success'] = True
        if self.step_count >= self.max_steps:
            terminated = True
            truncated = True
        return final_obs, reward, terminated, truncated, info

    def render(self):
        return self.unwrapped.render()
    
    def close(self):
        self.unwrapped.close()
from third_party.customized_box2d.HybridLunarLand import HybridLunarLander
def make_hybrid_lander(mode="learner",stack_frames=4,max_steps=1000):
    env = HybridLunarLander(mode=mode,render_mode="rgb_array")
    env = HybridLanderWrapper(env,stack_frames=stack_frames)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env




class HybridCarVectorWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4,max_steps=1000):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)
        self.max_steps = max_steps
        # Calculate new observation space size
        # Original Vector Dim = 14 (Phys) + 32 (Lidar) = 46
        # Stacked = 46 * 4 = 184
        base_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(base_dim * stack_frames,), 
            dtype=np.float32
        )
        self.env_category = 'box2d_hybrid_car_vector'

    def _get_flattened_obs(self):
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.stack_frames):
            self.frames.append(obs)
        return self._get_flattened_obs(), info

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_flattened_obs(), reward, terminated, truncated, info

from third_party.customized_box2d.HybridCar import HybridCarRacing
def make_hybrid_car_racing(mode="learner",stack_frames=4,max_steps=1000):
    env = HybridCarRacing(mode=mode,render_mode="rgb_array")
    env = HybridCarVectorWrapper(env,stack_frames=stack_frames,max_steps=max_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

class HybridWeatherVectorWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4,max_steps=2000):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)
        self.max_steps = max_steps
        base_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_dim * stack_frames,),
            dtype=np.float32,
        )
        self.env_category = 'box2d_hybrid_weather_vector'
    def _get_flattened_obs(self):
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.stack_frames):
            self.frames.append(obs)
        return self._get_flattened_obs(), info

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_flattened_obs(), reward, terminated, truncated, info

    def render(self):
        return self.unwrapped.render()

    def close(self):
        self.unwrapped.close()

from third_party.customized_box2d.HybridWeather import HybridWeather    
def make_hybrid_weather(mode="learner",stack_frames=4,max_steps=2000):
    assert mode in ["learner", "Sunny", "Rainy","Foggy"]
    env = HybridWeather(mode=mode,render_mode="rgb_array")
    env = HybridWeatherVectorWrapper(env,stack_frames=stack_frames,max_steps=max_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def _get_hydra_config_name() -> Optional[str]:
    """
    Best-effort retrieval of Hydra's `config_name` inside a @hydra.main entrypoint.

    Returns e.g. "dmc/CurrimaxAdv_cheetah" or "metaworld/simba_pickplace".
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        cfg = HydraConfig.get()
        name = getattr(getattr(cfg, "job", None), "config_name", None)
        return str(name) if name else None
    except Exception:
        return None


def _normalize_obs_config(obs_config: Any) -> Dict[str, Tuple[int, ...]]:
    """
    Convert an OmegaConf/DictConfig obs_config into a plain dict[str, tuple[int,...]].
    """
    if obs_config is None:
        return {}

    try:
        obs_config = OmegaConf.to_container(obs_config, resolve=True)
    except Exception:
        pass

    if obs_config is None:
        return {}
    if not isinstance(obs_config, dict):
        raise TypeError(f"obs_config must be a dict-like object, got {type(obs_config)}")

    out: Dict[str, Tuple[int, ...]] = {}
    for k, v in obs_config.items():
        if v is None:
            out[str(k)] = tuple()
        elif isinstance(v, tuple):
            out[str(k)] = tuple(int(x) for x in v)
        elif isinstance(v, list):
            out[str(k)] = tuple(int(x) for x in v)
        else:
            # Allow scalar shapes like 39
            out[str(k)] = (int(v),)
    return out


def _infer_env_category_from_config_name_or_cfg(
    config_name: Optional[str],
    cfg: Any,
) -> Literal["dmc", "metaworld", "weather", "box2d"]:
    """
    Infer which env creator to use.

    Priority:
      1) Hydra config_name prefix (e.g. "dmc/...", "metaworld/...")
      2) Inspect cfg.env keys
    """
    if config_name:
        prefix = config_name.split("/", 1)[0].strip().lower()
        if prefix in ("dmc", "metaworld", "myosuite", "box2d"):
            return prefix  # type: ignore[return-value]
        # Some users may pass config_name like "config/dmc/..." etc.
        lowered = config_name.lower()
        for p in ("dmc", "metaworld", "myosuite", "box2d"):
            if f"/{p}/" in lowered or lowered.startswith(f"{p}_") or lowered.startswith(f"{p}/"):
                return p  # type: ignore[return-value]

    env_cfg = getattr(cfg, "env", None)
    if env_cfg is not None:
        # DMC has (domain_name, task_name)
        if getattr(env_cfg, "domain_name", None) is not None and getattr(env_cfg, "task_name", None) is not None:
            return "dmc"
        # Box2D hybrid envs use env_name + mode/stack_frames
        if getattr(env_cfg, "box2d_env_name", None) is not None or getattr(env_cfg, "is_box2d", None):
            return "box2d"
        if getattr(env_cfg, "env_name", None) in ("bipedal", "lander", "racingcar", "car_racing", "weather", "hybrid_weather"):
            return "box2d"
        if getattr(env_cfg, "mode", None) is not None and getattr(env_cfg, "stack_frames", None) is not None:
            return "box2d"
        # Metaworld has env_name
        if getattr(env_cfg, "env_name", None) is not None:
            return "metaworld"
        # Myosuite typically uses env_name too, but projects often name it differently; handle explicitly if present.
        if getattr(env_cfg, "myosuite_env_name", None) is not None or getattr(env_cfg, "is_myosuite", None):
            return "myosuite"
       
    raise ValueError(
        f"Could not infer env category from config_name={config_name!r} or cfg.env keys. "
        f"Please name configs under dmc/ or metaworld/ (etc), or add recognizable keys to cfg.env."
    )


def make_env_and_eval_env_from_cfg(
    cfg: Any,
    *,
    obs_config: Optional[Any] = None,
    config_name: Optional[str] = None,
):
    """
    Create (env, eval_env) using cfg + (optional) Hydra config_name inference.

    This lets training scripts stay backend-agnostic:
      env, eval_env, obs_config = make_env_and_eval_env_from_cfg(cfg)
    """
    if config_name is None:
        config_name = _get_hydra_config_name()

    env_category = _infer_env_category_from_config_name_or_cfg(config_name, cfg)
    obs_config_norm = _normalize_obs_config(obs_config if obs_config is not None else cfg.get("obs_config", {}))

    env_cfg = cfg.env
    seed = getattr(env_cfg, "seed", 0)

    if env_category == "dmc":
        render_size = getattr(env_cfg, "render_size", (128, 128))
        try:
            render_size = tuple(render_size)
        except Exception:
            pass
        n_sub_steps = getattr(env_cfg, "n_sub_steps", 1)

        env = make_dmc_env(
            domain=env_cfg.domain_name,
            task=env_cfg.task_name,
            obs_config=obs_config_norm,
            seed=seed,
            render_size=render_size,
            n_sub_steps=n_sub_steps,
        )
        eval_env = make_dmc_env(
            domain=env_cfg.domain_name,
            task=env_cfg.task_name,
            obs_config=obs_config_norm,
            seed=seed,
            render_size=render_size,
            n_sub_steps=n_sub_steps,
        )
        return env, eval_env, obs_config_norm

    if env_category == "metaworld":
        env = make_metaworld_env(
            env_name=env_cfg.env_name,
            obs_config=obs_config_norm,
            seed=seed,
            camera_to_render=getattr(env_cfg, "camera_to_render", "corner4"),
            reward_scale= env_cfg.reward_scale ,
            step_penality= env_cfg.step_penality,
            sparse_reward = env_cfg.sparse_reward
        )
        eval_env = make_metaworld_env(
            env_name=env_cfg.env_name,
            obs_config=obs_config_norm,
            seed=seed,
            camera_to_render=getattr(env_cfg, "camera_to_render", "corner4"),
            reward_scale=env_cfg.reward_scale,
            step_penality=env_cfg.step_penality,
            sparse_reward = env_cfg.sparse_reward
        )
        return env, eval_env, obs_config_norm


    if env_category == "box2d":
        env_name = getattr(env_cfg, "env_name", None) or getattr(env_cfg, "box2d_env_name", None)
        if env_name is None:
            raise ValueError("box2d env requires cfg.env.env_name (or cfg.env.box2d_env_name)")
        mode = getattr(env_cfg, "mode", "learner")
        stack_frames = getattr(env_cfg, "stack_frames", 4)
        max_steps = getattr(env_cfg, "max_steps", 1000)
        env_key = str(env_name).lower()

        if env_key in ("bipedal", "walker", "hybrid_walker"):
            env = make_hybrid_walker(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            eval_env = make_hybrid_walker(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            return env, eval_env, obs_config_norm
        if env_key in ("lander", "hybrid_lander"):
            env = make_hybrid_lander(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            eval_env = make_hybrid_lander(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            return env, eval_env, obs_config_norm
        if env_key in ("racingcar", "car_racing", "hybrid_car_racing"):
            env = make_hybrid_car_racing(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            eval_env = make_hybrid_car_racing(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            return env, eval_env, obs_config_norm
        if env_key in ("weather", "hybrid_weather", "hybrid_weather_car"):
            env = make_hybrid_weather(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            eval_env = make_hybrid_weather(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            return env, eval_env, obs_config_norm

        raise ValueError(f"Unsupported box2d env_name: {env_name}")

    # Should be unreachable due to Literal return type, but keep as guard.
    raise ValueError(f"Unsupported env category: {env_category}")
