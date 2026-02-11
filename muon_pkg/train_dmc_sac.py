#!/usr/bin/env python
import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure repo root is on path
_REPO_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
sys.path.append(_REPO_ROOT)

from algo.artifacts import build_artifact_paths, ensure_artifact_dirs, update_cfg_with_artifacts, write_run_manifest
from env.env_utils import build_obs_shape_from_obs_config, make_dmc_env, set_random_seed
from data_buffer.replay_buffer import ReplayBuffer
from muon_pkg.sac import SAC
from muon_pkg.toy_model import DoubleQCritic, SquashedGaussianActor


@hydra.main(config_path="configs", config_name="sac_cheetah", version_base=None)
def train_sac(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.training.seed)

    artifacts = build_artifact_paths(cfg)
    ensure_artifact_dirs(artifacts)
    update_cfg_with_artifacts(cfg, artifacts)

    obs_config = cfg.obs_config
    env = make_dmc_env(
        domain=cfg.env.domain_name,
        task=cfg.env.task_name,
        obs_config=obs_config,
        seed=cfg.env.seed,
        render_size=cfg.env.render_size,
        n_sub_steps=cfg.env.n_sub_steps,
    )
    eval_env = make_dmc_env(
        domain=cfg.env.domain_name,
        task=cfg.env.task_name,
        obs_config=obs_config,
        seed=cfg.env.seed,
        render_size=cfg.env.render_size,
        n_sub_steps=cfg.env.n_sub_steps,
    )

    act_dim = int(env.action_space.shape[0])
    obs_shape = build_obs_shape_from_obs_config(obs_config)

    # Inject env action bounds into actor config (for correct scaling back to env space).
    actor_args = OmegaConf.to_container(cfg.actor_args, resolve=True)
    actor_args["action_high"] = env.action_space.high.tolist()
    actor_args["action_low"] = env.action_space.low.tolist()

    actor = SquashedGaussianActor(obs_shape=obs_shape, action_dim=act_dim, model_args=actor_args).to(device)
    critic = DoubleQCritic(
        obs_shape=obs_shape,
        action_dim=act_dim,
        model_args=OmegaConf.to_container(cfg.critic_args, resolve=True),
    ).to(device)

    # Flatten training config to dict and attach full config
    args = OmegaConf.to_container(cfg.training, resolve=True)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    args["full_config"] = cfg_resolved

    # Resolve target entropy "auto"
    if isinstance(args.get("target_entropy", None), str) and str(args["target_entropy"]).lower() == "auto":
        args["target_entropy"] = -float(act_dim)

    replay_buffer = ReplayBuffer(
        replay_dir=cfg.training.replay_dir,
        observation_shape=obs_shape,
        action_dim=act_dim,
        capacity=cfg.training.capacity,
        discount=cfg.training.discount,
        device=device,
    )

    # Checkpoint dir and save config snapshot
    with open(os.path.join(cfg.training.checkpoint_dir, "cfg.yaml"), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)
    write_run_manifest(artifacts.run_dir, cfg_resolved, artifacts)

    agent = SAC(actor=actor, critic=critic, args=args, device=device)
    print("Starting training (SAC)...")
    agent.run(env, replay_buffer=replay_buffer, eval_env=eval_env)
    print("Training completed.")


if __name__ == "__main__":
    train_sac()


