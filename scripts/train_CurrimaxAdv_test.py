#!/usr/bin/env python
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure paths are correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algo.artifacts import (
    build_artifact_paths,
    ensure_artifact_dirs,
    update_cfg_with_artifacts,
    write_run_manifest,
)

from model.simba import (
    DeterministicSimbaPolicy,
    SimbaCritics,
    SimbaValues,
)

from algo.oracles.oracle_loading import build_oracle_modules_from_cfg


from env.env_utils import (
    make_env_and_eval_env_from_cfg,
    set_random_seed,
    build_obs_shape_from_obs_config,
)
from data_buffer.replay_buffer import (
    ReplayBuffer
)





@hydra.main(config_path="../config", config_name="box2d/CurrimaxAdv_weather_learner", version_base=None)
def train_CurrimaxAdv(cfg: DictConfig):
    from algo.oracles.CurrimaxAdv_test import CurrMaxAdv

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    set_random_seed(cfg.training.seed)

    # Artifact layout (run_name, dirs, wandb group)
    OmegaConf.set_struct(cfg, False)
    artifacts = build_artifact_paths(cfg)
    ensure_artifact_dirs(artifacts)
    update_cfg_with_artifacts(cfg, artifacts)

    # Env (DeepMind Control cartpole swingup with dict obs)
    env, eval_env, obs_config = make_env_and_eval_env_from_cfg(cfg)

    act_dim = env.action_space.shape[0]
    obs_shape = build_obs_shape_from_obs_config(obs_config)
    # Learner (actor/critic)
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
    learner_value = SimbaValues(
        obs_shape=obs_shape,
        action_dim=act_dim,
        model_args=cfg.rl_critic_args,
    ).to(device)
    
    # Oracles (actors/critics) - create N placeholder oracle policies/critics
    n_oracles = int(cfg.training.n_oracles)
    oracles_dict = cfg.oracles_dict
    assert len(oracles_dict) == n_oracles, "Number of oracles in oracles_dict must match n_oracles"
    default_oracle_actor_args = (
        cfg.get("oracle_actor_args", None)
        or cfg.get("mlp_based_oracle_args", None)
    )
    if default_oracle_actor_args is None:
        raise ValueError("Missing oracle actor args in config (oracle_actor_args/mlp_based_oracle_args).")
    default_oracle_critic_args = cfg.oracle_critic_args

    oracles_actors, oracles_critics, oracles_values = build_oracle_modules_from_cfg(
        cfg=cfg,
        obs_shape=obs_shape,
        act_dim=act_dim,
        device=device,
        oracles_dict=oracles_dict,
        default_oracle_actor_args=default_oracle_actor_args,
        default_oracle_critic_args=default_oracle_critic_args,
    )
        
        
    # Print trainable parameters (MB)
    trainable_params = sum(p.numel() for p in learner_actor.parameters() if p.requires_grad)
    trainable_params += sum(p.numel() for p in learner_critic.parameters() if p.requires_grad)
    trainable_params_mb = trainable_params / 1024 / 1024
    print(f"Trainable parameters: {trainable_params_mb:.2f} MB")

    # Flatten training config to dict and attach full config
    args = OmegaConf.to_container(cfg.training, resolve=True)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    args["full_config"] = cfg_resolved

    # Agent
    agent = CurrMaxAdv(
        n_oracles=n_oracles,
        oracles_actors=oracles_actors,
        oracles_critics=oracles_critics,
        oracles_values=oracles_values,
        learner_actor=learner_actor,
        learner_critic=learner_critic,
        learner_value=learner_value,
        args=args,
        device=device,
    )

    # Replay buffer (disk-based) and loader
    replay_buffer = ReplayBuffer(
        replay_dir=cfg.training.replay_dir,
        observation_shape=obs_shape,
        action_dim=act_dim,
        capacity=cfg.training.capacity,
        discount=cfg.training.discount,
        device=device.type,
        save_episodes_to_disk=cfg.training.save_episodes_to_disk,
    )

    # Checkpoint dir and save config snapshot + run manifest
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.training.checkpoint_dir, "cfg.yaml"), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)
    write_run_manifest(artifacts.run_dir, cfg_resolved, artifacts)

    # Train
    print("Starting training (CurrMaxAdv)...")
    agent.run(env, replay_buffer=replay_buffer,eval_env=eval_env)
    print("Training completed.")

if __name__ == "__main__":
    train_CurrimaxAdv()


