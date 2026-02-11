import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb

from algo.base_algo import BaseAlgo
from algo.algo_utils import (
    RunningMeanStd,
    ObservationNormalizer,
    RewardNormalizer,
    categorical_td_loss,
    reshape_into_blocks,
    save_eval_results_to_csv,
)
from model.simba import DeterministicSimbaPolicy, SimbaCritics
from model.simba_base import EPS, l2normalize_network
from data_buffer.replay_buffer import ReplayBuffer

class SIMBA(BaseAlgo):
    """
    SIMBA algorithm implementation, refactored to inherit `BaseAlgo` so we don't duplicate
    wandb/eval/video/checkpoint/progress-bar plumbing.
    """

    def __init__(
        self,
        rl_agent: DeterministicSimbaPolicy,
        rl_critic: SimbaCritics,
        args: Dict[str, Any],
        device: torch.device,
    ):
        # Ensure BaseAlgo-required fields exist (no defaults; fail-fast if missing keys)
        args = dict(args)
        args["device"] = device

        super().__init__(args=args)

        self.rl_agent = rl_agent.to(device)
        self.rl_critic = rl_critic.to(device)

        # SIMBA Parameters
        self.gamma = args["discount"]
        self.tau = args["tau"]

        # Exploration noise parameters (with decay)
        self.min_exploration_std = args["min_exploration_std"]
        self.max_exploration_std = args["max_exploration_std"]
        self.exploration_decay_steps = args["exploration_decay_steps"]  # can be None, but must be explicit in config
        self.last_exploration_std = self.max_exploration_std
        self.target_policy_std = args["target_policy_std"]
        self.policy_delay = args["policy_delay"]  # TD3-style delayed policy update

        # Categorical critic parameters
        self.num_bins = args["num_bins"]
        self.min_v = args["min_v"]
        self.max_v = args["max_v"]

        # PERF: cache categorical support bins for distributional TD loss
        # Avoids re-creating torch.linspace(...) every critic update.
        self._cat_bin_values = torch.linspace(self.min_v, self.max_v, self.num_bins, device=device, dtype=torch.float32)
        self._cat_delta_z = (self.max_v - self.min_v) / (self.num_bins - 1)

        # Normalization
        self.normalize_observations = args["normalize_observations"]
        self.normalize_rewards = args["normalize_rewards"]
        self.normalized_g_max = args["normalized_g_max"]
        self.normalize_network = args["normalize_network"]

        # Initialize normalizers (shared impl in algo_utils)
        self.obs_normalizer: Optional[ObservationNormalizer]
        if self.normalize_observations:
            obs_dim = int(self.rl_agent.obs_dim)
            self.obs_normalizer = ObservationNormalizer(obs_dim=obs_dim, dtype=np.float32, eps=float(EPS))
            # Back-compat: keep `obs_rms` attribute used by older checkpoints / code
            self.obs_rms: Optional[RunningMeanStd] = self.obs_normalizer.rms
        else:
            self.obs_normalizer = None
            self.obs_rms = None

        self.reward_normalizer: Optional[RewardNormalizer]
        if self.normalize_rewards:
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.gamma), normalized_g_max=float(self.normalized_g_max), dtype=np.float32, eps=float(EPS)
            )
            # Back-compat fields (used in checkpoint payloads)
            self.G = self.reward_normalizer.G
            self.G_rms = self.reward_normalizer.G_rms
            self.G_r_max = self.reward_normalizer.G_r_max
        else:
            self.reward_normalizer = None


        # Initialize optimizers
        self.actor_optimizer = self.rl_agent.set_optimizer(self.args["actor_lr"], self.args["weight_decay"])
        self.critic_optimizer = self.rl_critic.set_optimizer(self.args["critic_lr"], self.args["weight_decay"])

        # Compile functions for better performance if enabled
        if self.args["use_compile"]:
            self._compiled_compute_critic_loss = torch.compile(self._compute_critic_loss, fullgraph=True)
            self._compiled_compute_actor_loss = torch.compile(self._compute_actor_loss, fullgraph=True)
        else:
            self._compiled_compute_critic_loss = self._compute_critic_loss
            self._compiled_compute_actor_loss = self._compute_actor_loss

    # -----------------------
    # Normalization utilities (numpy obs only)
    # -----------------------
    def _normalize_obs_np(self, obs: np.ndarray) -> np.ndarray:
        """Normalize a single numpy observation (no batch dim)."""
        if self.obs_normalizer is None:
            return obs
        return self.obs_normalizer.normalize_np(obs)

    def _normalize_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize a batch of observations using running statistics (for training)."""
        if self.obs_normalizer is None:
            return obs
        return self.obs_normalizer.normalize_tensor(obs)

    def _scale_reward_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        """Scale rewards tensor using running return statistics (for training)."""
        if self.reward_normalizer is None:
            return rewards
        return self.reward_normalizer.scale_rewards_tensor(rewards)

    def _update_obs_normalizer(self, obs: np.ndarray):
        """Update observation normalizer with new observations (numpy)."""
        if self.obs_normalizer is None:
            return
        self.obs_normalizer.update(obs)

    def _update_reward_normalizer(self, reward: float, done: bool):
        """Update reward normalizer with new reward."""
        if self.reward_normalizer is None:
            return
        self.reward_normalizer.update(float(reward), bool(done))
        # Keep back-compat fields in sync
        self.G = self.reward_normalizer.G
        self.G_r_max = self.reward_normalizer.G_r_max

    # -----------------------
    # Action selection
    # -----------------------
    def get_exploration_std(self) -> float:
        """Linear decay from max_exploration_std to min_exploration_std."""
        max_steps = self.exploration_decay_steps
        if max_steps is None:
            max_steps = self.args["total_timesteps"]
        if self.global_step >= max_steps:
            return float(self.min_exploration_std)
        progress = self.global_step / max_steps
        return float(self.max_exploration_std + (self.min_exploration_std - self.max_exploration_std) * progress)

    @torch.no_grad()
    def get_training_action(self, raw_obs, **training_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        # We assume env returns numpy obs with the correct shape.
        obs_np = np.asarray(raw_obs, dtype=np.float32)
        if self.normalize_observations:
            obs_np = self._normalize_obs_np(obs_np)
        obs_in = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)

        std = self.get_exploration_std()
        self.last_exploration_std = std

        normalized_action = self.rl_agent.get_action(obs_in, std=std)
        final_action = normalized_action * self.rl_agent.action_scale + self.rl_agent.action_bias
        return final_action.squeeze(0).detach().cpu().numpy(), normalized_action.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def get_inference_action(self, raw_obs, **inference_kwargs) -> np.ndarray:
        obs_np = np.asarray(raw_obs, dtype=np.float32)
        if self.normalize_observations:
            obs_np = self._normalize_obs_np(obs_np)
        obs_in = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)

        normalized_action = self.rl_agent.get_eval_action(obs_in)
        final_action = normalized_action * self.rl_agent.action_scale + self.rl_agent.action_bias
        return final_action.squeeze(0).detach().cpu().numpy()

    # -----------------------
    # Training update
    # -----------------------
    def _extract_obs_np(self, obs: Any) -> np.ndarray:
        """Extract observation as a (1, D) numpy array."""
        v = np.asarray(obs, dtype=np.float32)
        return v.reshape(1, -1)

    def _compute_critic_loss(self, data: dict):
        """Compute categorical TD loss for critic."""
        with torch.no_grad():
            next_actions = self.rl_agent.get_target_action(data["next_observations"], self.target_policy_std)
            next_qs, next_q_infos = self.rl_critic.get_target_value_with_info(data["next_observations"], next_actions)

            # next_qs: [num_qs, B, 1]
            min_idx = next_qs.squeeze(-1).argmin(dim=0)  # (B,)
            stacked_log_probs = torch.stack([info["log_prob"] for info in next_q_infos], dim=0)
            next_q_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)]

        pred_qs, pred_q_infos = self.rl_critic.get_value_with_info(data["observations"], data["actions"])
        gamma_n = self.gamma ** self.args["nstep"]

        loss_total = 0.0
        for pred_info in pred_q_infos:
            loss = categorical_td_loss(
                pred_log_probs=pred_info["log_prob"],
                target_log_probs=next_q_log_probs,
                reward=data["rewards"],
                done=data["dones"],
                gamma=gamma_n,
                num_bins=self.num_bins,
                min_v=self.min_v,
                max_v=self.max_v,
                bin_values=self._cat_bin_values,
                delta_z=self._cat_delta_z,
            )
            loss_total = loss_total + loss.mean()

        q_mean = pred_qs.mean()
        return loss_total, q_mean

    def _compute_actor_loss(self, data: dict):
        """Compute deterministic policy gradient loss for actor."""
        actions = self.rl_agent.get_action(data["observations"], std=0.0)
        qs = self.rl_critic.get_value(data["observations"], actions)
        q = qs.mean(dim=0)
        return -q.mean()

    def update_critic(self, data: dict) -> Dict[str, float]:
        qf_loss, qf_mean = self._compiled_compute_critic_loss(data)
        self.critic_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        nn.utils.clip_grad_norm_(self.rl_critic.parameters(), self.args["max_grad_norm"])
        self.critic_optimizer.step()
        if self.normalize_network:
            l2normalize_network(self.rl_critic.critic)
        return {"values/qf_loss": float(qf_loss.item()), "values/qf_mean": float(qf_mean.item())}

    def update_actor(self, data: dict) -> Dict[str, float]:
        actor_loss = self._compiled_compute_actor_loss(data)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.rl_agent.parameters(), self.args["max_grad_norm"])
        self.actor_optimizer.step()
        if self.normalize_network:
            l2normalize_network(self.rl_agent.policy)
        return {"actor/actor_loss": float(actor_loss.item()), "actor/exploration_std": float(self.last_exploration_std)}

    def update(self) -> Dict[str, Any]:
        """Update policy and value networks."""
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        self.rl_agent.train()
        self.rl_critic.train()

        info: Dict[str, Any] = {}
        num_blocks = self.args["utd_ratio"]

        data = self.replay_buffer.sample(self.args["batch_size"])
        if self.normalize_observations:
            data["observations"] = self._normalize_obs_tensor(data["observations"])
            data["next_observations"] = self._normalize_obs_tensor(data["next_observations"])

        if self.normalize_rewards:
            data["rewards"] = self._scale_reward_tensor(data["rewards"])

        mini_batches = reshape_into_blocks(data, num_blocks)

        critic_info: Dict[str, Any] = {}
        actor_info: Dict[str, Any] = {}

        for i in range(num_blocks):
            sub_batch = {k: v[i] for k, v in mini_batches.items()} if isinstance(mini_batches, dict) else mini_batches
            self.update_step += 1

            critic_info = self.update_critic(sub_batch)
            self.rl_critic.update_target(self.tau)

            if self.update_step % self.policy_delay == 0:
                actor_info = self.update_actor(sub_batch)
                self.rl_agent.update_target(self.tau)

        info.update(critic_info)
        if self.update_step % self.policy_delay == 0:
            info.update(actor_info)

        return info

    # -----------------------
    # Buffer + run loop
    # -----------------------

    def run(self, env, replay_buffer:ReplayBuffer, eval_env=None):
        """
        Main training loop. Uses BaseAlgo for resume/wandb/progressbar/finalize.
        """
        self._pre_run(env, replay_buffer, eval_env=eval_env)

        env_obs, _ = self.env.reset(seed=self.args["seed"])

        while self.global_step < self.total_timesteps:
            self.global_step += 1

            # Update observation normalizer stats FIRST (raw observation)
            if self.normalize_observations:
                obs_arr = self._extract_obs_np(env_obs)
                self._update_obs_normalizer(obs_arr)

            with torch.inference_mode():
                final_action, normalized_action = self.get_training_action(env_obs, normalize=True)

            next_env_obs, reward, terminated, truncated, infos = self.env.step(final_action)

            if self.normalize_rewards:
                self._update_reward_normalizer(reward, terminated or truncated)

            if isinstance(infos, dict) and "episode" in infos:
                self.log_local_train_episode(
                    global_step=self.global_step,
                    episode_reward=infos["episode"]["r"],
                    episode_length=infos["episode"]["l"],
                )

            # Store UNNORMALIZED transition
            transition = {
                    "observations": env_obs,
                    "next_observations": next_env_obs,
                    "actions": normalized_action,
                    "rewards": reward,
                    "dones": terminated,
                    "truncated": truncated,
                }
            self.replay_buffer.add_transition(transition)

            env_obs = next_env_obs

            if self.global_step >= self.args["learning_starts"]:
                train_info = self.update()
                if self.use_wandb:
                    if isinstance(infos, dict) and "episode" in infos:
                        train_info["env/episode_reward"] = infos["episode"]["r"]
                    train_info["time/total_timesteps"] = self.global_step
                    train_info["time/fps"] = self.global_step / (time.time() - self.start_time)
                    wandb.log(train_info, step=self.global_step)

            self.progress_bar.update(1)

            if (self.global_step + 1) % self.eval_every == 0:
                _eval_env = eval_env if eval_env is not None else env
                eval_results = self.evaluate(_eval_env, num_episodes=self.args["eval_episodes"], normalize=True)
                save_eval_results_to_csv(
                    self.args["checkpoint_dir"],
                    self.args.get("eval_run_name", self.args["run_name"]),
                    eval_results,
                    eval_dir=self.args.get("eval_dir"),
                )
                if self.use_wandb:
                    for k, v in eval_results.items():
                        wandb.log({f"eval/{k}": v}, step=self.global_step)

            if (self.global_step + 1) % self.save_freq == 0:
                save_path = f"{self.args['checkpoint_dir']}/{self.args['run_name']}_{self.global_step}.pt"
                self.save_model(save_path)

            if truncated or terminated:
                env_obs, _ = self.env.reset()

        return self._after_run()

    # -----------------------
    # BaseAlgo checkpoint hooks
    # -----------------------
    def _save_model_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint.update(
            {
                "rl_agent_state_dict": self.rl_agent.state_dict(),
                "rl_critic_state_dict": self.rl_critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            }
        )

        if self.obs_normalizer is not None:
            # Keep existing key name for compatibility
            checkpoint["obs_rms"] = self.obs_normalizer.rms.state_dict()
        if self.reward_normalizer is not None:
            checkpoint["reward_norm"] = {
                "G": self.reward_normalizer.G,
                "G_rms": self.reward_normalizer.G_rms.state_dict(),
                "G_r_max": self.reward_normalizer.G_r_max,
            }

        return checkpoint

    def _load_model_checkpoint(self, checkpoint: dict) -> dict:
        if "rl_agent_state_dict" in checkpoint:
            self.rl_agent.load_state_dict(checkpoint["rl_agent_state_dict"])
        if "rl_critic_state_dict" in checkpoint:
            self.rl_critic.load_state_dict(checkpoint["rl_critic_state_dict"])

        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        if "obs_rms" in checkpoint and self.obs_normalizer is not None:
            # legacy payload: RunningMeanStd dict
            self.obs_normalizer.rms.load_state_dict(checkpoint["obs_rms"])
        if "reward_norm" in checkpoint and self.reward_normalizer is not None:
            self.reward_normalizer.load_state_dict(checkpoint["reward_norm"])
            # Keep back-compat fields in sync
            self.G = self.reward_normalizer.G
            self.G_r_max = self.reward_normalizer.G_r_max

        return checkpoint


