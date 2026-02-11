#!/usr/bin/env python
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure paths are correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from model.simba import (
    DeterministicSimbaPolicy,
    SimbaCritics,
    SimbaValues,
)

from model.mlp import DeterministicPolicy
from model.simba_share import SharedSimbaCritics, SharedSimbaValues


from env.env_utils import (
    make_env_and_eval_env_from_cfg,
    set_random_seed,
    build_obs_shape_from_obs_config,
    _is_missing_checkpoint,
    _banner,
)
from data_buffer.replay_buffer import (
    ReplayBuffer
)





@hydra.main(config_path="../config", config_name="dmc/SharedmaxAdv_cheetah", version_base=None)
def train_SharedmaxAdv(cfg: DictConfig):
    from algo.oracles.SharedmaxAdv import SharedMaxAdv

    def _load_oracle_model_args(ckpt_path: str):
        """
        Load per-oracle overrides from sibling cfg.yaml (same directory as checkpoint).
        Returns (oracle_actor_cfg, oracle_critic_cfg) where either may be None.
        """
        file_dir = os.path.dirname(ckpt_path)
        cfg_path = os.path.join(file_dir, "cfg.yaml")
        if not os.path.exists(cfg_path):
            return None
        oracle_cfg = OmegaConf.load(cfg_path)
        return oracle_cfg.actor_args

    def _build_oracle_actor(*, obs_shape, act_dim, device, actor_args):
        actor = DeterministicPolicy(
            obs_shape=obs_shape,
            action_dim=act_dim,
            model_args=actor_args,
        ).to(device)
        return actor

    def _load_oracle_checkpoint(*, name: str, ckpt_path: str, device, actor):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        if "actor_state_dict" in checkpoint:
            actor.load_state_dict(checkpoint["actor_state_dict"], strict=False)
            print(f"Loaded oracle actor {name} from {ckpt_path}")
        else:
            _banner(f"Checkpoint {ckpt_path} does not contain actor_state_dict")
        for p in actor.parameters():
            p.requires_grad = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    set_random_seed(cfg.training.seed)

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
    
    # Oracles (actors only)
    n_oracles = int(cfg.training.n_oracles)
    oracles_dict = cfg.oracles_dict
    assert len(oracles_dict) == n_oracles, "Number of oracles in oracles_dict must match n_oracles"
    oracles_actors = {}
    default_oracle_actor_args = cfg.oracle_actor_args

    for name, ckpt_path in oracles_dict.items():
        ## Null checkpoint, we use random oracle 
        if _is_missing_checkpoint(ckpt_path):
            actor = _build_oracle_actor(
                obs_shape=obs_shape,
                act_dim=act_dim,
                device=device,
                actor_args=default_oracle_actor_args,
            )
            oracles_actors[name] = actor
            # Random oracle is still not trained by this script/algorithm.
            for p in actor.parameters():
                p.requires_grad = False
            _banner(f"Use Random Oracle: {name}")
            continue
        
        oracle_actor_cfg = _load_oracle_model_args(ckpt_path)
        actor_args = oracle_actor_cfg if oracle_actor_cfg is not None else default_oracle_actor_args

        actor = _build_oracle_actor(
            obs_shape=obs_shape,
            act_dim=act_dim,
            device=device,
            actor_args=actor_args,
        )
        oracles_actors[name] = actor

        print(f"Oracle {name} model_args: actor={actor_args}")
        _load_oracle_checkpoint(
            name=name,
            ckpt_path=ckpt_path,
            device=device,
            actor=actor,
        )
        
    # Shared critic/value for per-policy estimation (heads: learner + each oracle)
    shared_critic_args = cfg.oracle_critic_args
    shared_value_args = cfg.oracle_critic_args

    n_heads = int(n_oracles) + 1
    shared_critic_args = OmegaConf.merge(shared_critic_args, {"n_heads": n_heads})
    shared_value_args = OmegaConf.merge(shared_value_args, {"n_heads": n_heads})

    shared_critic = SharedSimbaCritics(
        obs_shape=obs_shape,
        action_dim=act_dim,
        model_args=shared_critic_args,
    ).to(device)
    shared_value = SharedSimbaValues(
        obs_shape=obs_shape,
        action_dim=act_dim,
        model_args=shared_value_args,
    ).to(device)
        
        
    # Print trainable parameters (MB)
    trainable_params = sum(p.numel() for p in learner_actor.parameters() if p.requires_grad)
    trainable_params += sum(p.numel() for p in learner_critic.parameters() if p.requires_grad)
    trainable_params += sum(p.numel() for p in learner_value.parameters() if p.requires_grad)
    trainable_params += sum(p.numel() for p in shared_critic.parameters() if p.requires_grad)
    trainable_params += sum(p.numel() for p in shared_value.parameters() if p.requires_grad)
    trainable_params_mb = trainable_params / 1024 / 1024
    print(f"Trainable parameters: {trainable_params_mb:.2f} MB")

    # Flatten training config to dict and attach full config
    args = OmegaConf.to_container(cfg.training, resolve=True)
    args["full_config"] = OmegaConf.to_container(cfg, resolve=True)

    # Agent
    agent = SharedMaxAdv(
        n_oracles=n_oracles,
        oracles_actors=oracles_actors,
        shared_critic=shared_critic,
        shared_value=shared_value,
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

    # Checkpoint dir and save config snapshot
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.training.checkpoint_dir, "cfg.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    # Train
    print("Starting training (SharedMaxAdv)...")
    agent.run(env, replay_buffer=replay_buffer, eval_env=eval_env)
    print("Training completed.")

if __name__ == "__main__":
    train_SharedmaxAdv()


