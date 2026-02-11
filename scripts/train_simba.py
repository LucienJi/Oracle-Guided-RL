#!/usr/bin/env python
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure paths are correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from algo.artifacts import build_artifact_paths, ensure_artifact_dirs, update_cfg_with_artifacts, write_run_manifest
from model.simba import DeterministicSimbaPolicy, SimbaCritics
from env.env_utils import make_env_and_eval_env_from_cfg, set_random_seed, build_obs_shape_from_obs_config
from data_buffer.replay_buffer import ReplayBuffer





@hydra.main(config_path="../config", config_name="dmc/simba_cartpole_sparse", version_base=None)
def train_simba(cfg: DictConfig):
    from algo.baselines.simba import SIMBA
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    set_random_seed(cfg.training.seed)


    # Artifact layout (run_name, dirs, wandb group)
    artifacts = build_artifact_paths(cfg)
    ensure_artifact_dirs(artifacts)
    update_cfg_with_artifacts(cfg, artifacts)

    env, eval_env, obs_config = make_env_and_eval_env_from_cfg(cfg)
    act_dim = env.action_space.shape[0]
    obs_shape = build_obs_shape_from_obs_config(obs_config)

    # Create residual actor critic model
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
    
    ## Print trainable parameters
    trainable_params = sum(p.numel() for p in rl_agent.parameters() if p.requires_grad) + sum(p.numel() for p in rl_critic.parameters() if p.requires_grad)
    trainable_params_mb = trainable_params / 1024 / 1024
    print(f"Trainable parameters: {trainable_params_mb:.2f} MB")
        

    # Flatten training config to dict and attach full config
    args = OmegaConf.to_container(cfg.training, resolve=True)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    args["full_config"] = cfg_resolved

    # Agent
    agent = SIMBA(
        rl_agent=rl_agent,
        rl_critic=rl_critic,
        args=args,
        device=device
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
    with open(os.path.join(cfg.training.checkpoint_dir, "cfg.yaml"), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)
    write_run_manifest(artifacts.run_dir, cfg_resolved, artifacts)

    # Train
    print("Starting training (MaxAdv)...")
    agent.run(env, replay_buffer=replay_buffer,eval_env=eval_env)
    print("Training completed.")



if __name__ == "__main__":
    train_simba()


