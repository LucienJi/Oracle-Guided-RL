import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
torch.set_float32_matmul_precision('high')
from algo.base_algo import BaseAlgo
from algo.algo_utils import ObservationNormalizer, reshape_into_blocks, save_eval_results_to_csv
from data_buffer.replay_buffer import ReplayBuffer
from model.mlp import OraclePolicyBase
from model.simba import DeterministicSimbaPolicy


class BC(BaseAlgo):
    """
    Minimal offline Behavior Cloning baseline.

    - Trains from a disk-backed `ReplayBuffer` via uniform sampling (`ReplayBuffer.sample`).
    - Uses a simple MSE loss between the policy's eval action (normalized) and dataset actions.
    - Inherits from `BaseAlgo` for logging/eval/checkpointing utilities.
    """

    def __init__(
        self,
        bc_agent: OraclePolicyBase,
        args: Dict[str, Any],
        device: torch.device,
        *,
        expert_policy: Optional[DeterministicSimbaPolicy] = None,
        expert_obs_normalizer: Optional[ObservationNormalizer] = None,
    ):
        args = dict(args)
        args["device"] = device
        super().__init__(args=args)

        self.bc_agent = bc_agent.to(device)
        self.expert_policy = expert_policy.to(device) if expert_policy is not None else None
        if self.expert_policy is not None:
            self.expert_policy.eval()
            for p in self.expert_policy.parameters():
                p.requires_grad_(False)
        self.expert_obs_normalizer = expert_obs_normalizer


        # Optimizer
        weight_decay = self.args["weight_decay"]
        self.actor_optimizer = self.bc_agent.set_optimizer(self.args["actor_lr"], weight_decay=weight_decay)

        # Training knobs
        self.utd_ratio = self.args["utd_ratio"]
        self.max_grad_norm = self.args["max_grad_norm"]

        # Optional torch.compile (keep off by default)
        self.use_compile = self.args["use_compile"]
        self._compiled_compute_actor_loss = self._compute_actor_loss
        if self.use_compile and hasattr(torch, "compile"):
            try:
                self._compiled_compute_actor_loss = torch.compile(self._compute_actor_loss, fullgraph=True)
            except Exception as e:
                print(f"[BC] torch.compile failed: {e}. Falling back to eager.")

    # -----------------------
    # Action selection
    # -----------------------
    @torch.no_grad()
    def get_training_action(self, raw_obs, **training_kwargs):
        # BC is offline; keep a reasonable default if someone uses it for rollout.
        return self.get_inference_action(raw_obs)

    @torch.no_grad()
    def get_inference_action(self, raw_obs, **inference_kwargs) -> np.ndarray:
        obs_np = np.asarray(raw_obs, dtype=np.float32)
        obs_in = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)  # (1, D)

        normalized_action = self.bc_agent.get_eval_action(obs_in)
        final_action = normalized_action * self.bc_agent.action_scale + self.bc_agent.action_bias
        return final_action.squeeze(0).detach().cpu().numpy()

    # -----------------------
    # Training update
    # -----------------------
    def _compute_actor_loss(self, data: dict) -> torch.Tensor:
        observations = data["observations"]  # (B, D)
        pred_actions = self.bc_agent.get_eval_action(observations)

        # Default: supervise from dataset actions (assumed normalized in [-1, 1])
        if self.expert_policy is None:
            actions = data["actions"]
            return F.mse_loss(pred_actions, actions)

        # Expert supervision: query expert on sampled observations.
        with torch.no_grad():
            obs_for_expert = observations
            if self.expert_obs_normalizer is not None:
                obs_for_expert = self.expert_obs_normalizer.normalize_tensor(obs_for_expert)
            expert_actions = self.expert_policy.get_eval_action(obs_for_expert)

        return F.mse_loss(pred_actions, expert_actions)

    def update_actor(self, data: dict) -> Dict[str, float]:
        actor_loss = self._compiled_compute_actor_loss(data)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.bc_agent.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        return {"losses/actor_loss": float(actor_loss.item())}

    def update(self) -> Dict[str, Any]:
        self.bc_agent.train()

        info: Dict[str, Any] = {}
        data = self.replay_buffer.sample(self.args["batch_size"])

        # Optional UTD: split the sampled batch into `utd_ratio` blocks and do multiple updates.
        num_blocks = int(self.utd_ratio)
        mini_batches = reshape_into_blocks(data, num_blocks) if num_blocks > 1 else data

        for i in range(num_blocks):
            sub_batch = {k: v[i] for k, v in mini_batches.items()} if num_blocks > 1 else mini_batches
            self.update_step += 1
            info = self.update_actor(sub_batch)

        return info

    # -----------------------
    # Run loop (offline)
    # -----------------------
    @torch.no_grad()
    def _evaluate_expert(self, env, num_episodes: int = 10) -> Dict[str, float]:
        if self.expert_policy is None:
            return {}

        returns = []
        lengths = []
        for ep in range(int(num_episodes)):
            obs, _ = env.reset(seed=int(self.args["seed"]) + ep)
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                obs_np = np.asarray(obs, dtype=np.float32)
                if self.expert_obs_normalizer is not None:
                    obs_np = self.expert_obs_normalizer.normalize_np(obs_np)
                obs_t = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)
                norm_act = self.expert_policy.get_eval_action(obs_t)
                act = norm_act * self.expert_policy.action_scale + self.expert_policy.action_bias
                obs, r, terminated, truncated, _info = env.step(act.squeeze(0).cpu().numpy())
                done = bool(terminated or truncated)
                ep_ret += float(r)
                ep_len += 1
            returns.append(ep_ret)
            lengths.append(ep_len)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
        }

    def run(self, env, replay_buffer: ReplayBuffer, eval_env=None):
        """
        Offline BC training loop.
        - `replay_buffer` is assumed to already be preloaded from disk via ReplayBuffer(replay_dir=...).
        - `env` is only used for evaluation (if desired).
        """
        self._pre_run(env, replay_buffer, eval_env=eval_env)

        # Evaluate expert once before BC training starts (if provided).
        if self.expert_policy is not None:
            _eval_env = eval_env if eval_env is not None else env
            expert_results = self._evaluate_expert(_eval_env, num_episodes=self.eval_episodes)
            print(f"[BC] Expert eval: {expert_results}")
            if self.use_wandb:
                wandb.log({f"expert/{k}": v for k, v in expert_results.items()}, step=self.global_step)

        while self.global_step < self.total_timesteps:
            self.global_step += 1

            train_info = self.update()
            if self.use_wandb:
                train_info["time/total_timesteps"] = self.global_step
                train_info["time/fps"] = self.global_step / (time.time() - self.start_time)
                wandb.log(train_info, step=self.global_step)

            self.progress_bar.update(1)

            if (self.global_step + 1) % self.eval_every == 0:
                _eval_env = eval_env if eval_env is not None else env
                eval_results = self.evaluate(_eval_env, num_episodes=self.eval_episodes)
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

        return self._after_run()

    # -----------------------
    # BaseAlgo checkpoint hooks
    # -----------------------
    def _save_model_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint.update(
            {
                "actor_state_dict": self.bc_agent.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            }
        )
        return checkpoint

    def _load_model_checkpoint(self, checkpoint: dict) -> dict:
        if "actor_state_dict" in checkpoint:
            self.bc_agent.load_state_dict(checkpoint["actor_state_dict"])
        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        return checkpoint