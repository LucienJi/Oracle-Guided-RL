#!/usr/bin/env python
import os
import sys
import glob
from types import SimpleNamespace
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure paths are correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from algo.artifacts import build_artifact_paths, ensure_artifact_dirs, update_cfg_with_artifacts, write_run_manifest
from model.mlp import DeterministicPolicy
from model.simba import DeterministicSimbaPolicy
from model.simba_base import EPS
from env.env_utils import make_env_and_eval_env_from_cfg, set_random_seed, build_obs_shape_from_obs_config
from data_buffer.replay_buffer import ReplayBuffer
from algo.algo_utils import ObservationNormalizer





@hydra.main(config_path="../config", config_name="dmc/bc_cheetah", version_base=None)
def train_bc(cfg: DictConfig):
    from algo.baselines.bc import BC
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    set_random_seed(cfg.training.seed)

    # Artifact layout (run_name, dirs, wandb group)
    artifacts = build_artifact_paths(cfg)
    ensure_artifact_dirs(artifacts)
    update_cfg_with_artifacts(cfg, artifacts)

    # Env (DeepMind Control cartpole swingup with dict obs)
    raw_obs_config = cfg.get("obs_config", {})
    obs_config = {k: tuple(v) for k, v in raw_obs_config.items()}
    obs_shape = build_obs_shape_from_obs_config(obs_config)
    input_mapping = cfg.get("input_mapping", {})
    
    print("obs_config:",obs_config)
    print("input_mapping:",input_mapping)

    env, eval_env,obs_config = make_env_and_eval_env_from_cfg(cfg)

    act_dim = env.action_space.shape[0]

    # Create residual actor critic model
    actor = DeterministicPolicy(
        obs_shape=obs_shape,
        action_dim=act_dim,
        model_args=cfg.actor_args,
    )
    
    
    ## Print trainable parameters
    trainable_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    trainable_params_mb = trainable_params / 1024 / 1024
    print(f"Trainable parameters: {trainable_params_mb:.2f} MB")
        

    # Flatten training config to dict and attach full config
    args = OmegaConf.to_container(cfg.training, resolve=True)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    args["full_config"] = cfg_resolved

    # Optional: expert policy (SIMBA) for supervision
    expert_policy = None
    expert_obs_normalizer = None
    expert_dir = cfg.training.get("expert_dir", None)
    if expert_dir:
        expert_cfg_path = cfg.training.get("expert_cfg_path", os.path.join(expert_dir, "cfg.yaml"))
        expert_ckpt_path = cfg.training.get("expert_ckpt_path", None)
        if expert_ckpt_path is None:
            best_matches = sorted(glob.glob(os.path.join(expert_dir, "*_best.pt")))
            if len(best_matches) > 0:
                expert_ckpt_path = best_matches[-1]
            else:
                pt_matches = sorted(glob.glob(os.path.join(expert_dir, "*.pt")))
                if len(pt_matches) == 0:
                    raise FileNotFoundError(f"No .pt checkpoints found in expert_dir={expert_dir}")
                expert_ckpt_path = pt_matches[-1]

        expert_cfg = OmegaConf.load(expert_cfg_path)
        expert_args_dict = OmegaConf.to_container(expert_cfg.rl_agent_args, resolve=True)
        # Ensure action bounds match the env (if supported by the policy)
        expert_args_dict["action_high"] = env.action_space.high.tolist()
        expert_args_dict["action_low"] = env.action_space.low.tolist()
        expert_model_args = SimpleNamespace(**expert_args_dict)

        expert_policy = DeterministicSimbaPolicy(
            obs_shape=obs_shape,
            action_dim=act_dim,
            model_args=expert_model_args,
        ).to(device)
        expert_policy.eval()

        ckpt = torch.load(expert_ckpt_path, map_location=device, weights_only=False)
        if "rl_agent_state_dict" not in ckpt:
            raise KeyError(f"Expected key 'rl_agent_state_dict' in expert checkpoint: {expert_ckpt_path}")
        expert_policy.load_state_dict(ckpt["rl_agent_state_dict"])

        if "obs_rms" in ckpt:
            expert_obs_normalizer = ObservationNormalizer(obs_dim=int(obs_shape[0]), eps=float(EPS))
            expert_obs_normalizer.load_state_dict(ckpt["obs_rms"])

        print(f"Loaded expert from cfg={expert_cfg_path}, ckpt={expert_ckpt_path}")
        print(f"Expert obs normalizer loaded: {expert_obs_normalizer is not None}")

    # Agent
    agent = BC(
        bc_agent=actor,
        args=args,
        device=device,
        expert_policy=expert_policy,
        expert_obs_normalizer=expert_obs_normalizer,
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
    print("Starting training (BC)...")
    agent.run(env, replay_buffer=replay_buffer,eval_env=eval_env)
    print("Training completed.")



if __name__ == "__main__":
    train_bc()


