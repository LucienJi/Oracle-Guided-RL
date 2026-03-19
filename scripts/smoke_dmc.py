#!/usr/bin/env python
"""Run a lightweight DMC submission smoke test."""

import json
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algo.artifacts import build_artifact_paths, ensure_artifact_dirs, update_cfg_with_artifacts
from algo.oracles.oracle_loading import build_oracle_modules_from_cfg
from env.env_utils import build_obs_shape_from_obs_config, make_env_and_eval_env_from_cfg, set_random_seed
from model.simba import DeterministicSimbaPolicy, SimbaCritics, SimbaValues


def _validate_smoke_cfg(cfg: DictConfig) -> None:
    smoke_cfg = cfg.get("smoke", None)
    if smoke_cfg is None:
        raise ValueError("Smoke config requires a top-level 'smoke' section.")

    workflow = str(smoke_cfg.get("workflow", "")).strip().lower()
    if workflow not in {"simba", "currimaxadv"}:
        raise ValueError(f"Unsupported smoke.workflow '{workflow}'. Expected 'simba' or 'currimaxadv'.")

    num_steps = int(smoke_cfg.get("num_steps", 0))
    if num_steps <= 0:
        raise ValueError(f"smoke.num_steps must be positive, got {num_steps}.")

    if not cfg.get("obs_config", {}):
        raise ValueError("Smoke config requires a non-empty obs_config.")


def _select_eval_action(actor: DeterministicSimbaPolicy, obs: np.ndarray) -> np.ndarray:
    device = next(actor.parameters()).device
    obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        normalized_action = actor.get_eval_action(obs_tensor)
        final_action = normalized_action * actor.action_scale + actor.action_bias
    return final_action.squeeze(0).cpu().numpy()


def _count_trainable_params_mb(*modules) -> float:
    total = 0
    for module in modules:
        total += sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total / 1024 / 1024


def run_smoke(cfg: DictConfig) -> dict:
    """Build the configured DMC workflow and execute a few inference steps."""
    _validate_smoke_cfg(cfg)

    workflow = str(cfg.smoke.workflow).strip().lower()
    num_steps = int(cfg.smoke.num_steps)
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
            raise ValueError(f"Smoke path expects a 1D continuous action space, got shape={action_shape!r}.")

        obs_shape = build_obs_shape_from_obs_config(obs_config)
        act_dim = action_shape[0]

        learner_actor = DeterministicSimbaPolicy(
            obs_shape=obs_shape,
            action_dim=act_dim,
            model_args=cfg.rl_agent_args,
        ).to(device)
        learner_critic = SimbaCritics(
            obs_shape=obs_shape,
            action_dim=act_dim,
            model_args=cfg.rl_critic_args,
        ).to(device)

        oracle_names = []
        if workflow == "currimaxadv":
            learner_value = SimbaValues(
                obs_shape=obs_shape,
                action_dim=act_dim,
                model_args=cfg.rl_critic_args,
            ).to(device)
            _ = learner_value  # Constructed intentionally for smoke coverage.
            default_oracle_actor_args = (
                cfg.get("oracle_actor_args", None)
                or cfg.get("mlp_based_oracle_args", None)
            )
            if default_oracle_actor_args is None:
                raise ValueError("CurrimaxAdv smoke config must provide oracle actor args.")
            oracles_actors, _, _ = build_oracle_modules_from_cfg(
                cfg=cfg,
                obs_shape=obs_shape,
                act_dim=act_dim,
                device=device,
                oracles_dict=cfg.oracles_dict,
                default_oracle_actor_args=default_oracle_actor_args,
                default_oracle_critic_args=cfg.oracle_critic_args,
            )
            oracle_names = sorted(oracles_actors.keys())

        obs, _ = env.reset(seed=cfg.training.seed)
        _, _ = eval_env.reset(seed=cfg.training.seed + 1)

        cumulative_reward = 0.0
        executed_steps = 0
        terminated = False
        truncated = False
        while executed_steps < num_steps and not (terminated or truncated):
            action = _select_eval_action(learner_actor, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += float(reward)
            executed_steps += 1

        summary = {
            "workflow": workflow,
            "domain_name": cfg.env.domain_name,
            "task_name": cfg.env.task_name,
            "obs_shape": list(obs_shape),
            "action_dim": int(act_dim),
            "executed_steps": executed_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "cumulative_reward": cumulative_reward,
            "artifact_run_dir": artifacts.run_dir,
            "checkpoint_dir": cfg.training.checkpoint_dir,
            "replay_dir": cfg.training.replay_dir,
            "oracle_names": oracle_names,
            "trainable_params_mb": round(_count_trainable_params_mb(learner_actor, learner_critic), 2),
        }
        return summary
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


@hydra.main(config_path="../config", config_name="dmc/submission_smoke_cartpole", version_base=None)
def main(cfg: DictConfig) -> None:
    summary = run_smoke(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
