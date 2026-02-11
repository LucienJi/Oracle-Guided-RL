import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from algo.base_algo import BaseAlgo
from algo.algo_utils import RunningMeanStd, reshape_into_blocks, save_eval_results_to_csv
from data_buffer.replay_buffer import ReplayBuffer
from muon_pkg.optim_utils import MomentumSchedule, build_optimizer_stack
from muon_pkg.toy_model import CriticWithTarget, DoubleQCritic, SquashedGaussianActor


class SAC(BaseAlgo):
    """
    Soft Actor-Critic (SAC) with a flexible optimizer stack to test Muon vs Adam/AdamW,
    momentum scheduling, and selective Muon application on first/last layers.
    """

    def __init__(
        self,
        actor: SquashedGaussianActor,
        critic: DoubleQCritic,
        args: Dict[str, Any],
        device: torch.device,
    ):
        args = dict(args)
        args["device"] = device
        super().__init__(args=args)

        self.device = device
        self.actor = actor.to(device)
        self.critic = CriticWithTarget(critic.to(device))

        # Core SAC params
        self.gamma = float(self.args["discount"])
        self.tau = float(self.args["tau"])
        self.policy_delay = int(self.args["policy_delay"])
        self.max_grad_norm = float(self.args["max_grad_norm"])
        self.utd_ratio = int(self.args["utd_ratio"])

        # Normalization
        self.normalize_observations = bool(self.args["normalize_observations"])
        self.normalize_rewards = bool(self.args["normalize_rewards"])
        self.normalized_g_max = float(self.args["normalized_g_max"])
        self.obs_rms: Optional[RunningMeanStd]
        if self.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=(1, int(self.actor.obs_dim)), dtype=np.float32)
        else:
            self.obs_rms = None

        # Reward normalization (simple return-based scaler like SIMBA)
        self.G = 0.0
        self.G_rms = RunningMeanStd(shape=(1,), dtype=np.float32)
        self.G_r_max = 0.0

        # Entropy / alpha
        self.autotune_alpha = bool(self.args["autotune_alpha"])
        self.target_entropy = float(self.args["target_entropy"])
        init_alpha = float(self.args["init_alpha"])
        if self.autotune_alpha:
            self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(init_alpha, device=device, dtype=torch.float32)))
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.log_alpha = None
            self.alpha = float(init_alpha)

        # Optimizers (actor/critic)
        actor_opt_cfg = dict(self.args["actor_optimizer"])
        critic_opt_cfg = dict(self.args["critic_optimizer"])
        self.actor_optimizers = build_optimizer_stack(self.actor, cfg=actor_opt_cfg, role="actor")
        self.critic_optimizers = build_optimizer_stack(self.critic.critic, cfg=critic_opt_cfg, role="critic")
        

        # Alpha optimizer (always Adam)
        alpha_lr = float(self.args["alpha_lr"])
        if self.autotune_alpha:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, foreach=True)
        else:
            self.alpha_optimizer = None

        # Momentum schedules (optional)
        self.actor_momentum_schedule = MomentumSchedule.from_cfg(self.args["actor_momentum_schedule"])
        self.critic_momentum_schedule = MomentumSchedule.from_cfg(self.args["critic_momentum_schedule"])

        # Optional warmup alias for delayed momentum (requested in experiment design).
        warmup_steps = int(self.args["warmup_steps"])
        warmup_reset = bool(self.args["warmup_reset_state"])
        if warmup_steps > 0:
            if not bool(self.actor_momentum_schedule.enabled):
                actor_opt_case = str(actor_opt_cfg["case"]).lower()
                if actor_opt_case == "muon":
                    actor_default = float(actor_opt_cfg["muon"]["momentum"])
                else:
                    actor_default = float(actor_opt_cfg["betas"][0])
                self.actor_momentum_schedule = MomentumSchedule(
                    enabled=True,
                    step_unit="global_step",
                    start_step=warmup_steps,
                    value=float(actor_default),
                    reset_state_on_start=warmup_reset,
                )
            if not bool(self.critic_momentum_schedule.enabled):
                critic_opt_case = str(critic_opt_cfg["case"]).lower()
                if critic_opt_case == "muon":
                    critic_default = float(critic_opt_cfg["muon"]["momentum"])
                else:
                    critic_default = float(critic_opt_cfg["betas"][0])
                self.critic_momentum_schedule = MomentumSchedule(
                    enabled=True,
                    step_unit="global_step",
                    start_step=warmup_steps,
                    value=float(critic_default),
                    reset_state_on_start=warmup_reset,
                )


        # Compile (optional)
        self.use_compile = bool(self.args["use_compile"])
        self._compiled_compute_critic_losses = self._compute_critic_losses
        self._compiled_compute_actor_losses = self._compute_actor_losses
        if self.use_compile and hasattr(torch, "compile"):
            self._compiled_compute_critic_losses = torch.compile(self._compute_critic_losses, fullgraph=True)
            self._compiled_compute_actor_losses = torch.compile(self._compute_actor_losses, fullgraph=True)

    # -----------------------
    # Normalization helpers
    # -----------------------
    def _normalize_obs_np(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return obs
        mean = self.obs_rms.mean.squeeze()
        std = np.sqrt(self.obs_rms.var.squeeze() + 1e-8)
        return (obs - mean) / std

    def _normalize_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_rms is None:
            return obs
        mean = torch.from_numpy(self.obs_rms.mean.squeeze()).float().to(obs.device)
        std = torch.sqrt(torch.from_numpy(self.obs_rms.var.squeeze()).float().to(obs.device) + 1e-8)
        return (obs - mean) / std

    def _update_obs_normalizer(self, obs_np: np.ndarray) -> None:
        if self.obs_rms is None:
            return
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)
        self.obs_rms.update(obs_np)

    def _update_reward_normalizer(self, reward: float, done: bool) -> None:
        if not self.normalize_rewards:
            return
        self.G = self.gamma * (1 - done) * self.G + reward
        self.G_rms.update(np.array([[self.G]]))
        self.G_r_max = max(self.G_r_max, abs(self.G))

    def _scale_reward_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.normalize_rewards:
            return rewards
        var_val = self.G_rms.var[0] if isinstance(self.G_rms.var, np.ndarray) else self.G_rms.var
        var_denominator = float(np.sqrt(var_val + 1e-8))
        min_required_denominator = self.G_r_max / self.normalized_g_max if self.normalized_g_max > 0 else 0.0
        denominator = max(var_denominator, float(min_required_denominator), 1e-8)
        return rewards / denominator

    # -----------------------
    # Action selection
    # -----------------------
    @torch.no_grad()
    def get_training_action(self, raw_obs, **training_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        obs_np = np.asarray(raw_obs, dtype=np.float32)
        if self.normalize_observations:
            obs_np = self._normalize_obs_np(obs_np)
        obs = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)

        a_norm, _logp = self.actor.sample(obs)
        a_env = a_norm * self.actor.action_scale + self.actor.action_bias
        return a_env.squeeze(0).cpu().numpy(), a_norm.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def get_inference_action(self, raw_obs, **inference_kwargs) -> np.ndarray:
        obs_np = np.asarray(raw_obs, dtype=np.float32)
        if self.normalize_observations:
            obs_np = self._normalize_obs_np(obs_np)
        obs = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)

        a_norm = self.actor.eval_action(obs)
        a_env = a_norm * self.actor.action_scale + self.actor.action_bias
        return a_env.squeeze(0).cpu().numpy()

    # -----------------------
    # Losses
    # -----------------------
    def _compute_critic_losses(self, data: dict) -> Dict[str, torch.Tensor]:
        obs = data["observations"]
        next_obs = data["next_observations"]
        act = data["actions"]
        rew = data["rewards"].view(-1, 1)
        discounts = data["discounts"].view(-1, 1)

        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            tq1, tq2 = self.critic.target(next_obs, next_a)
            tmin = torch.min(tq1, tq2)
            target_v = tmin - float(self.alpha) * next_logp
            target_q = rew + discounts * target_v

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        out = {
            "critic_loss": critic_loss,
            "q1_mean": q1.mean(),
            "q2_mean": q2.mean(),
        }
        return out

    def _compute_actor_losses(self, data: dict) -> Dict[str, torch.Tensor]:
        """
        Compute actor (and optional alpha) losses.

        IMPORTANT: This must be run AFTER the critic optimizer step for the same minibatch.
        Otherwise, the actor loss graph may reference critic weights that were modified in-place
        by the critic update, which triggers an autograd versioning error.
        """
        obs = data["observations"]

        a_pi, logp = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (float(self.alpha) * logp - q_pi).mean()

        out = {
            "actor_loss": actor_loss,
            "logp_mean": logp.mean(),
        }
        if self.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            out["alpha_loss"] = alpha_loss
        return out

    def _maybe_apply_momentum_schedules(self) -> Dict[str, float]:
        info: Dict[str, float] = {}
        # Choose step unit per schedule
        def _step_for(s: MomentumSchedule) -> int:
            if s.step_unit == "update_step":
                return int(self.update_step)
            return int(self.global_step)

        # Default values come from the optimizer config (what we'd use with no schedule).
        actor_opt_cfg = dict(self.args["actor_optimizer"])
        critic_opt_cfg = dict(self.args["critic_optimizer"])

        actor_case = str(actor_opt_cfg["case"]).lower()
        critic_case = str(critic_opt_cfg["case"]).lower()

        actor_default = float(actor_opt_cfg["muon"]["momentum"]) if actor_case == "muon" else float(actor_opt_cfg["betas"][0])
        critic_default = float(critic_opt_cfg["muon"]["momentum"]) if critic_case == "muon" else float(critic_opt_cfg["betas"][0])

        v_a = self.actor_momentum_schedule.maybe_apply(
            self.actor_optimizers,
            step=_step_for(self.actor_momentum_schedule),
            default_value=actor_default,
        )
        v_c = self.critic_momentum_schedule.maybe_apply(
            self.critic_optimizers,
            step=_step_for(self.critic_momentum_schedule),
            default_value=critic_default,
        )
        if v_a is not None:
            info["opt/actor_momentum_or_beta1"] = float(v_a)
        if v_c is not None:
            info["opt/critic_momentum_or_beta1"] = float(v_c)
        return info

    def _zero_grad_stack(self, opts):
        for opt in opts:
            opt.zero_grad(set_to_none=True)

    def _step_stack(self, opts):
        for opt in opts:
            opt.step()

    def update(self) -> Dict[str, Any]:
        self.actor.train()
        self.critic.train()

        # Sample batch
        data = self.replay_buffer.sample(self.args["batch_size"])
        if self.normalize_observations:
            data["observations"] = self._normalize_obs_tensor(data["observations"])
            data["next_observations"] = self._normalize_obs_tensor(data["next_observations"])
        if self.normalize_rewards:
            data["rewards"] = self._scale_reward_tensor(data["rewards"])

        # UTD: split batch into blocks
        num_blocks = int(self.utd_ratio)
        mini_batches = reshape_into_blocks(data, num_blocks) if num_blocks > 1 else data

        info: Dict[str, Any] = {}
        for i in range(num_blocks):
            sub = {k: v[i] for k, v in mini_batches.items()} if num_blocks > 1 else mini_batches
            self.update_step += 1
            info.update(self._maybe_apply_momentum_schedules())

            critic_losses = self._compiled_compute_critic_losses(sub)

            # Critic update
            self._zero_grad_stack(self.critic_optimizers)
            critic_losses["critic_loss"].backward()
            nn.utils.clip_grad_norm_(self.critic.critic.parameters(), self.max_grad_norm)
            self._step_stack(self.critic_optimizers)

            # Actor + alpha update (delayed optional)
            if self.update_step % self.policy_delay == 0:
                actor_losses = self._compiled_compute_actor_losses(sub)
                # Single backward pass to avoid "backward through the graph a second time"
                # under torch.compile / AOTAutograd when actor_loss and alpha_loss come from
                # the same forward.
                self._zero_grad_stack(self.actor_optimizers)
                if self.autotune_alpha and self.alpha_optimizer is not None:
                    self.alpha_optimizer.zero_grad(set_to_none=True)

                total_actor_loss = actor_losses["actor_loss"]
                if self.autotune_alpha and self.alpha_optimizer is not None:
                    total_actor_loss = total_actor_loss + actor_losses["alpha_loss"]

                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self._step_stack(self.actor_optimizers)

                if self.autotune_alpha and self.alpha_optimizer is not None:
                    self.alpha_optimizer.step()
                    self.alpha = float(self.log_alpha.exp().item())

                # Soft update target after actor update for stability
                self.critic.update_target(self.tau)

            # Metrics (overwrite with last mini-batch values)
            info.update(
                {
                    "losses/critic_loss": float(critic_losses["critic_loss"].item()),
                    "values/q1_mean": float(critic_losses["q1_mean"].item()),
                    "values/q2_mean": float(critic_losses["q2_mean"].item()),
                    "policy/alpha": float(self.alpha),
                }
            )
            if self.update_step % self.policy_delay == 0:
                info["losses/actor_loss"] = float(actor_losses["actor_loss"].item())
                info["policy/logp_mean"] = float(actor_losses["logp_mean"].item())
                if self.autotune_alpha and "alpha_loss" in actor_losses:
                    info["losses/alpha_loss"] = float(actor_losses["alpha_loss"].item())
        return info

    # -----------------------
    # Run loop
    # -----------------------
    def run(self, env, replay_buffer: ReplayBuffer, eval_env=None):
        self._pre_run(env, replay_buffer, eval_env=eval_env)
        env_obs, _ = self.env.reset(seed=self.args["seed"])

        learning_starts = int(self.args["learning_starts"])

        while self.global_step < self.total_timesteps:
            self.global_step += 1

            # Update obs normalization with raw obs
            if self.normalize_observations:
                self._update_obs_normalizer(np.asarray(env_obs, dtype=np.float32))

            # Action selection
            if self.global_step < learning_starts:
                final_action = self.env.action_space.sample()
                # Convert to normalized [-1, 1] for storage (best-effort)
                # If env space is already [-1,1], this is exact.
                ah = np.asarray(self.actor.action_high.cpu().numpy())
                al = np.asarray(self.actor.action_low.cpu().numpy())
                action_scale = (ah - al) / 2.0
                action_bias = (ah + al) / 2.0
                normalized_action = (np.asarray(final_action) - action_bias) / (action_scale + 1e-8)
                normalized_action = np.clip(normalized_action, -1.0, 1.0).astype(np.float32)
            else:
                with torch.inference_mode():
                    final_action, normalized_action = self.get_training_action(env_obs)

            next_env_obs, reward, terminated, truncated, infos = self.env.step(final_action)

            if self.normalize_rewards:
                self._update_reward_normalizer(float(reward), bool(terminated or truncated))

            transition = {
                "observations": env_obs,
                "next_observations": next_env_obs,
                "actions": normalized_action,
                "rewards": reward,
                "dones": terminated,
                "truncated": truncated,
                "global_step": int(self.global_step),
            }
            self.replay_buffer.add_transition(transition)
            env_obs = next_env_obs

            # Updates
            if self.global_step >= learning_starts:
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
                eval_results = self.evaluate(_eval_env, num_episodes=self.args["eval_episodes"])
                save_eval_results_to_csv(self.args["checkpoint_dir"], self.args["run_name"], eval_results)
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
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.critic.state_dict(),
                "critic_target_state_dict": self.critic.critic_target.state_dict(),
                "actor_optimizers_state_dict": [opt.state_dict() for opt in self.actor_optimizers],
                "critic_optimizers_state_dict": [opt.state_dict() for opt in self.critic_optimizers],
                "alpha": float(self.alpha),
            }
        )
        if self.autotune_alpha and self.log_alpha is not None:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            if self.alpha_optimizer is not None:
                checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        if self.obs_rms is not None:
            checkpoint["obs_rms"] = self.obs_rms.state_dict()
        if self.normalize_rewards:
            checkpoint["reward_norm"] = {
                "G": self.G,
                "G_rms": self.G_rms.state_dict(),
                "G_r_max": self.G_r_max,
            }
        return checkpoint

    def _load_model_checkpoint(self, checkpoint: dict) -> dict:
        if "actor_state_dict" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
        if "critic_state_dict" in checkpoint:
            self.critic.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "critic_target_state_dict" in checkpoint:
            self.critic.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])

        if "actor_optimizers_state_dict" in checkpoint:
            for opt, st in zip(self.actor_optimizers, checkpoint["actor_optimizers_state_dict"]):
                opt.load_state_dict(st)
        if "critic_optimizers_state_dict" in checkpoint:
            for opt, st in zip(self.critic_optimizers, checkpoint["critic_optimizers_state_dict"]):
                opt.load_state_dict(st)

        if "alpha" in checkpoint:
            self.alpha = float(checkpoint["alpha"])
        if self.autotune_alpha and self.log_alpha is not None and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
            self.alpha = float(self.log_alpha.exp().item())
            if self.alpha_optimizer is not None and "alpha_optimizer_state_dict" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])

        if "obs_rms" in checkpoint and self.obs_rms is not None:
            self.obs_rms.load_state_dict(checkpoint["obs_rms"])
        if "reward_norm" in checkpoint and self.normalize_rewards:
            self.G = checkpoint["reward_norm"]["G"]
            self.G_rms.load_state_dict(checkpoint["reward_norm"]["G_rms"])
            self.G_r_max = checkpoint["reward_norm"]["G_r_max"]
        return checkpoint


