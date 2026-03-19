#!/usr/bin/env python
"""Train a SIMBA policy on a Hydra-configured environment."""

import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure the repository root is importable when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algo.artifacts import build_artifact_paths, ensure_artifact_dirs, update_cfg_with_artifacts, write_run_manifest
from data_buffer.replay_buffer import ReplayBuffer
from env.env_utils import build_obs_shape_from_obs_config, make_env_and_eval_env_from_cfg, set_random_seed
from model.simba import DeterministicSimbaPolicy, SimbaCritics


def _count_trainable_params_mb(*modules) -> float:
    total = 0
    for module in modules:
        total += sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total / 1024 / 1024


def _require_positive_int(value, field_name: str) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{field_name} must be positive, got {value_int}.")
    return value_int


def _validate_simba_cfg(cfg: DictConfig) -> None:
    env_cfg = cfg.env
    if getattr(env_cfg, "domain_name", None) is None or getattr(env_cfg, "task_name", None) is None:
        raise ValueError("SIMBA training requires cfg.env.domain_name and cfg.env.task_name for the DMC submission path.")

    obs_config = cfg.get("obs_config", {})
    if not obs_config:
        raise ValueError("SIMBA training requires a non-empty obs_config.")

    training_cfg = cfg.training
    for field_name in ("total_timesteps", "capacity", "batch_size", "eval_every", "save_freq"):
        _require_positive_int(training_cfg[field_name], f"training.{field_name}")

    discount = float(training_cfg.discount)
    if not 0.0 < discount <= 1.0:
        raise ValueError(f"training.discount must be in (0, 1], got {discount}.")


def _write_cfg_snapshot(cfg: DictConfig) -> None:
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.training.checkpoint_dir, "cfg.yaml"), "w", encoding="utf-8") as handle:
        OmegaConf.save(cfg, handle, resolve=True)


@hydra.main(config_path="../config", config_name="dmc/submission_smoke_cartpole", version_base=None)
def train_simba(cfg: DictConfig) -> None:
    """Entry point for SIMBA training."""
    from algo.baselines.simba import SIMBA

    _validate_simba_cfg(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.training.seed)

    artifacts = build_artifact_paths(cfg)
    ensure_artifact_dirs(artifacts)
    update_cfg_with_artifacts(cfg, artifacts)

    env = None
    eval_env = None
    try:
        env, eval_env, obs_config = make_env_and_eval_env_from_cfg(cfg)
        action_shape = getattr(env.action_space, "shape", None)
        if not action_shape or len(action_shape) != 1:
            raise ValueError(f"SIMBA expects a 1D continuous action space, got shape={action_shape!r}.")

        act_dim = action_shape[0]
        obs_shape = build_obs_shape_from_obs_config(obs_config)
        if len(obs_shape) != 1 or obs_shape[0] <= 0:
            raise ValueError(f"Failed to build a flat observation shape from obs_config: {obs_shape!r}")

        rl_agent = DeterministicSimbaPolicy(
            obs_shape=obs_shape,
            action_dim=act_dim,
            model_args=cfg.rl_agent_args,
        )
        rl_critic = SimbaCritics(
            obs_shape=obs_shape,
            action_dim=act_dim,
            model_args=cfg.rl_critic_args,
        )

        print(f"Trainable parameters: {_count_trainable_params_mb(rl_agent, rl_critic):.2f} MB")

        args = OmegaConf.to_container(cfg.training, resolve=True)
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(args, dict) or not isinstance(cfg_resolved, dict):
            raise TypeError("Expected resolved Hydra configs to convert to dictionaries.")
        args["full_config"] = cfg_resolved

        agent = SIMBA(
            rl_agent=rl_agent,
            rl_critic=rl_critic,
            args=args,
            device=device,
        )

        replay_buffer = ReplayBuffer(
            replay_dir=cfg.training.replay_dir,
            observation_shape=obs_shape,
            action_dim=act_dim,
            capacity=cfg.training.capacity,
            discount=cfg.training.discount,
            device=device.type,
            save_episodes_to_disk=cfg.training.save_episodes_to_disk,
        )

        _write_cfg_snapshot(cfg)
        write_run_manifest(artifacts.run_dir, cfg_resolved, artifacts)

        print("Starting training (SIMBA)...")
        agent.run(env, replay_buffer=replay_buffer, eval_env=eval_env)
        print("Training completed.")
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    train_simba()
