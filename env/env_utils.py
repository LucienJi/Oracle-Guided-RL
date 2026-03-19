import importlib
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import random
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf


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


def _optional_dependency_error(module_name: str, feature_name: str, install_hint: str) -> ImportError:
    return ImportError(
        f"{feature_name} requires the optional dependency '{module_name}'. {install_hint}"
    )


def _import_optional_module(module_name: str, feature_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise _optional_dependency_error(module_name, feature_name, install_hint) from exc


def _apply_dm_control_mujoco_compat():
    """Patch dm_control's MjModel and MjData field lists to match current MuJoCo bindings.
    Newer MuJoCo (e.g. 3.6) renamed/removed fields. Drop or alias sizes so struct_indexer
    only requests attributes that exist on the installed mujoco structs.
    """
    try:
        import mujoco
        from dm_control.mujoco.wrapper import mjbindings
        sizes = mjbindings.sizes
        minimal_xml = "<mujoco><worldbody><body><geom type='sphere' size='.1'/></body></worldbody></mujoco>"
        model = mujoco.MjModel.from_xml_string(minimal_xml)
        data = mujoco.MjData(model)
        valid_mjmodel = {a for a in dir(model) if not a.startswith("_")}
        valid_mjdata = {a for a in dir(data) if not a.startswith("_")}
        renames_model = [("cam_orthographic", "cam_projection")]
        mjmodel = sizes.array_sizes.get("mjmodel")
        if mjmodel is not None:
            for old_name, new_name in renames_model:
                if old_name in mjmodel and new_name not in mjmodel and new_name in valid_mjmodel:
                    mjmodel[new_name] = mjmodel.pop(old_name)
            for key in list(mjmodel.keys()):
                if key not in valid_mjmodel:
                    mjmodel.pop(key, None)
        mjdata = sizes.array_sizes.get("mjdata")
        if mjdata is not None:
            for key in list(mjdata.keys()):
                if key not in valid_mjdata:
                    mjdata.pop(key, None)
    except Exception:
        pass


def _require_dm_control_suite():
    _apply_dm_control_mujoco_compat()
    return _import_optional_module(
        "dm_control.suite",
        "DeepMind Control environments",
        "Install the DMC dependencies from environment.yml before running DMC training or smoke tests.",
    )


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
        suite = _require_dm_control_suite()
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


def make_env_and_eval_env_from_cfg(
    cfg: Any,
    *,
    obs_config: Optional[Any] = None,
    config_name: Optional[str] = None,
):
    """
    Create (env, eval_env, obs_config_norm) for DMC using cfg.

    Expects cfg.env.domain_name and cfg.env.task_name. Other simulators have been removed.
    """
    obs_config_norm = _normalize_obs_config(obs_config if obs_config is not None else cfg.get("obs_config", {}))

    env_cfg = cfg.env
    if getattr(env_cfg, "domain_name", None) is None or getattr(env_cfg, "task_name", None) is None:
        raise ValueError("DMC env requires cfg.env.domain_name and cfg.env.task_name")

    seed = getattr(env_cfg, "seed", 0)
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
