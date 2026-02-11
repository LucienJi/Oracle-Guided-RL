import random
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

torch.set_float32_matmul_precision("high")

from algo.base_algo import BaseAlgo
from algo.algo_utils import ObservationNormalizer, RewardNormalizer, RunningMeanStd, categorical_td_loss, reshape_into_blocks, save_eval_results_to_csv
from data_buffer.replay_buffer import ReplayBuffer
from model.mlp import OraclePolicyBase
from model.simba import DeterministicSimbaPolicy, SimbaCritics, SimbaValues
from model.simba_base import EPS, l2normalize_network
from model.simba_share import SharedSimbaCritics, SharedSimbaValues


class SharedMaxAdv(BaseAlgo):
    """
    Refactored "shared-head" MaxAdv.

    Key design:
    - A *separate* learner critic/value (`learner_critic`, `learner_value`) is used for
      critic training and the actor's RL loss (TD3-style term).
    - A *shared-head* critic/value (`shared_critic`, `shared_value`) is used for
      per-policy Q/V estimation for action scoring/selection during interaction.

    Head mapping convention:
      - head_idx=0: learner
      - head_idx=i+1: oracle i in `self.oracles_names` order
    """

    def __init__(
        self,
        n_oracles: int,
        oracles_actors: Dict[str, OraclePolicyBase],
        shared_critic: SharedSimbaCritics,
        shared_value: SharedSimbaValues,
        learner_actor: DeterministicSimbaPolicy,
        learner_critic: SimbaCritics,
        learner_value: SimbaValues,
        args: Dict[str, Any],
        device: torch.device,
    ):
        args = dict(args)
        args["device"] = device
        super().__init__(args=args)

        self.device = device
        self.n_oracles = int(n_oracles)

        self.oracles_actors = dict(oracles_actors)
        self.oracles_names = list(self.oracles_actors.keys())
        self.oracles_2_index = {name: i for i, name in enumerate(self.oracles_names)}
        assert len(self.oracles_names) == self.n_oracles

        # Move models to device + build list views for hot path
        for name in self.oracles_names:
            self.oracles_actors[name] = self.oracles_actors[name].to(device)
        self._oracle_actor_list = [self.oracles_actors[name] for name in self.oracles_names]

        # Shared-head estimators (for interaction scoring)
        self.shared_critic = shared_critic.to(device)
        self.shared_value = shared_value.to(device)

        # Separate learner critic/value (for training + actor RL loss)
        self.learner_actor = learner_actor.to(device)
        self.learner_critic = learner_critic.to(device)
        self.learner_value = learner_value.to(device)

        # -----------------------
        # Hyperparameters
        # -----------------------
        self.oracle_std_multiplier = float(self.args["oracle_std_multiplier"])

        # Exploration / sampling
        self.num_proposals = int(self.args["num_proposals"])
        self.sampling_max_temperature = float(self.args["sampling_max_temperature"])
        self.sampling_min_temperature = float(self.args["sampling_min_temperature"])
        self.kappa = float(self.args["kappa"])

        # Actor loss weighting
        self.beta = float(self.args["beta"])

        # TD3-style / scheduling
        self.l1_loss_weight = float(self.args["l1_loss_weight"])
        self.rl_loss_weight = float(self.args["rl_loss_weight"])
        self.use_adaptive_actor_loss_weight = bool(self.args["use_adaptive_actor_loss_weight"])
        self.learner_learning_starts = int(self.args["learner_learning_starts"])
        self.policy_delay = int(self.args["policy_delay"])
        self.task_specific_noise = float(self.args["task_specific_noise"])
        self.clip_noise = float(self.args["clip_noise"])

        # Distributional critic settings
        self.num_bins = int(self.args["num_bins"])
        self.min_v = float(self.args["min_v"])
        self.max_v = float(self.args["max_v"])
        self.gamma = float(self.args["discount"])

        # Exploration noise for behavior policy
        self.min_exploration_noise = float(self.args["min_exploration_noise"])
        self.max_exploration_noise = float(self.args["max_exploration_noise"])
        self.target_policy_noise = float(self.args["target_policy_noise"])

        # Normalization settings (SIMBA-style)
        self.normalize_observations = bool(self.args.get("normalize_observations", True))
        self.normalize_rewards = bool(self.args.get("normalize_rewards", True))
        self.normalized_g_max = float(self.args.get("normalized_g_max", 10.0))

        # Oracle warmup schedule
        self.per_oracle_explore_steps = int(self.args["per_oracle_explore_steps"])
        self.total_explore_steps = int(self.per_oracle_explore_steps * (self.n_oracles + 1))

        # Logging / controls
        self.log_every = int(self.args.get("log_every", 10))
        self.action_selection_buffer = deque(maxlen=int(self.args["action_selection_buffer_size"]))

        # Baselines / toggles
        self.baseline_greedy_guided = bool(self.args.get("baseline_greedy_guided", False))
        # Shared networks are intended for interaction scoring/selection.
        # Keep actor-weight computation on the separate learner critic/value unless explicitly enabled.
        self.use_shared_for_advantage = bool(self.args.get("use_shared_for_advantage", False))

        # Mixed precision / compilation
        self.use_compile = bool(self.args.get("use_compile", False))

        # -----------------------
        # Head sanity checks
        # -----------------------
        needed_heads = self.n_oracles + 1
        shared_heads = int(getattr(self.shared_critic, "n_heads", 1))
        if shared_heads < needed_heads:
            raise ValueError(
                f"Shared critic has n_heads={shared_heads}, but need at least {needed_heads} (= n_oracles+1)."
            )
        shared_heads_v = int(getattr(self.shared_value, "n_heads", 1))
        if shared_heads_v < needed_heads:
            raise ValueError(
                f"Shared value has n_heads={shared_heads_v}, but need at least {needed_heads} (= n_oracles+1)."
            )

        # -----------------------
        # Normalizers
        # -----------------------
        self.obs_normalizer = None
        self.obs_rms = None
        if self.normalize_observations:
            obs_dim = int(self.learner_actor.obs_dim)
            self.obs_normalizer = ObservationNormalizer(obs_dim=obs_dim, dtype=np.float32, eps=float(EPS))
            # Back-compat: checkpoint payload key expects `obs_rms`
            self.obs_rms = self.obs_normalizer.rms

        self.reward_normalizer = None
        if self.normalize_rewards:
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.gamma), normalized_g_max=float(self.normalized_g_max), dtype=np.float32, eps=float(EPS)
            )
            # Back-compat fields (used in checkpoint payloads)
            self.G = self.reward_normalizer.G
            self.G_rms = self.reward_normalizer.G_rms
            self.G_r_max = self.reward_normalizer.G_r_max

        # -----------------------
        # Optimizers
        # -----------------------
        actor_wd = float(self.args["actor_weight_decay"])
        critic_wd = float(self.args["critic_weight_decay"])
        value_wd = float(self.args["value_weight_decay"])
        self.actor_optimizer = self.learner_actor.set_optimizer(self.args["actor_lr"], weight_decay=actor_wd)
        self.critic_optimizer = self.learner_critic.set_optimizer(self.args["critic_lr"], weight_decay=critic_wd)
        self.value_optimizer = self.learner_value.set_optimizer(self.args["value_lr"], weight_decay=value_wd)

        self.shared_critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.shared_value_optimizer: Optional[torch.optim.Optimizer] = None
        self.shared_critic_optimizer = self.shared_critic.set_optimizer(self.args["critic_lr"], weight_decay=critic_wd)
        self.shared_value_optimizer = self.shared_value.set_optimizer(self.args["value_lr"], weight_decay=value_wd)

        # -----------------------
        # Compilation hooks
        # -----------------------
        self._compiled_compute_learner_critic_loss = self._compute_learner_critic_loss
        self._compiled_compute_shared_head_critic_loss = self._compute_shared_head_critic_loss
        self._compiled_compute_learner_actor_loss = self._compute_learner_actor_loss

        if self.use_compile and hasattr(torch, "compile"):
            self._compiled_compute_learner_critic_loss = self._try_compile(
                self._compiled_compute_learner_critic_loss, "learner_critic_loss", fullgraph=True
            )
            self._compiled_compute_shared_head_critic_loss = self._try_compile(
                self._compiled_compute_shared_head_critic_loss, "shared_head_critic_loss", fullgraph=True
            )
            self._compiled_compute_learner_actor_loss = self._try_compile(
                self._compiled_compute_learner_actor_loss, "learner_actor_loss", fullgraph=True
            )
            # NOTE: action sampling uses torch.distributions (torch.Size) -> allow graph breaks
            self._get_learner_training_action = self._try_compile(
                self._get_learner_training_action, "get_learner_training_action", fullgraph=False
            )
            self._get_oracle_training_action = self._try_compile(
                self._get_oracle_training_action, "get_oracle_training_action", fullgraph=False
            )
            self.prepare_action_candidates_obs = self._try_compile(
                self.prepare_action_candidates_obs, "prepare_action_candidates_obs", fullgraph=False
            )

        print(f"SharedMaxAdv: compile={'on' if self.use_compile else 'off'}")

    # -----------------------
    # torch.compile helper
    # -----------------------
    def _try_compile(self, fn, name: str, *, fullgraph: bool = True):
        try:
            return torch.compile(fn, fullgraph=fullgraph)
        except Exception as e:
            print(f"[SharedMaxAdv] torch.compile failed for {name}: {e}. Falling back to eager.")
            return fn

    # -----------------------
    # Noise / temperature schedules
    # -----------------------
    def get_action_noise(self, global_step: int, max_steps: int, min_noise: float, max_noise: float) -> float:
        """Linear decay from max_noise to min_noise."""
        if global_step > max_steps:
            return float(min_noise)
        epsilon = float(global_step) / float(max_steps)
        return float(max_noise + (min_noise - max_noise) * epsilon)

    def add_noise_to_action(self, action: torch.Tensor, noise: float, clip_noise: float) -> torch.Tensor:
        """Add clipped Gaussian noise to action tensor (normalized action space)."""
        eps = torch.randn_like(action)
        eps.mul_(noise)
        eps.clamp_(-clip_noise, clip_noise)
        action = action.add(eps)
        return action.clamp_(-1.0, 1.0)

    def get_temperature(self, current_step: int, max_steps: int) -> float:
        """Get temperature for stochastic sampling with linear decay."""
        if current_step >= max_steps:
            return float(self.sampling_min_temperature)
        progress = float(current_step) / float(max_steps)
        return float(self.sampling_max_temperature + (self.sampling_min_temperature - self.sampling_max_temperature) * progress)

    # -----------------------
    # Normalization (SIMBA-style)
    # -----------------------
    def _normalize_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
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
    # Replay sampling helpers
    # -----------------------
    def _prepare_sampled_batch(self, batch: dict) -> dict:
        """Apply observation/reward normalization to a sampled batch."""
        if self.normalize_observations:
            batch["observations"] = self._normalize_obs_tensor(batch["observations"])
            batch["next_observations"] = self._normalize_obs_tensor(batch["next_observations"])
        if self.normalize_rewards:
            batch["rewards"] = self._scale_reward_tensor(batch["rewards"])
        return batch

    # -----------------------
    # Tensor helpers
    # -----------------------
    def _obs_to_tensor(self, raw_obs: np.ndarray) -> torch.Tensor:
        obs_np = np.asarray(raw_obs, dtype=np.float32)
        return torch.from_numpy(obs_np).to(self.device).unsqueeze(0)  # (1, D)

    @staticmethod
    def _flatten_action_candidates(actions: torch.Tensor) -> torch.Tensor:
        """Flatten sampled actions from shape (k, B, A) to (k*B, A)."""
        if actions.dim() == 2:
            return actions
        if actions.dim() == 3:
            k, b, a = actions.shape
            return actions.reshape(k * b, a)
        raise ValueError(f"Unexpected action tensor shape: {tuple(actions.shape)}")

    def _oracle_name_from_head_idx(self, head_idx: int) -> str:
        if head_idx <= 0:
            raise ValueError("head_idx=0 is reserved for learner")
        return self.oracles_names[int(head_idx) - 1]

    # -----------------------
    # Interaction (policy + critics)
    # -----------------------
    @torch.no_grad()
    def _get_learner_training_action(
        self, obs_t: torch.Tensor, noise_std: float, k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_action = (
            self.learner_actor.get_action(obs_t, noise_std)
            if k == 1
            else self.learner_actor.sample_action(obs_t, noise_std, k)
        )
        final_action = normalized_action * self.learner_actor.action_scale + self.learner_actor.action_bias
        return final_action, normalized_action

    @torch.no_grad()
    def _get_oracle_training_action(
        self, obs_t: torch.Tensor, oracle_index: int, k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        oracle_name = self.oracles_names[int(oracle_index)]
        actor = self.oracles_actors[oracle_name]
        normalized_action = (
            actor.get_action(obs_t, self.oracle_std_multiplier)
            if k == 1
            else actor.sample_action(obs_t, self.oracle_std_multiplier, k)
        )
        final_action = normalized_action * actor.action_scale + actor.action_bias
        return final_action, normalized_action

    @torch.no_grad()
    def _get_shared_head_training_critic(
        self, obs_t: torch.Tensor, actions: torch.Tensor, head_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared estimator used for scoring.
        Returns q_mean, q_std, v (each shape: (B,))
        """
        qs = self.shared_critic.get_value_single_head(obs_t, actions, head_idx=head_idx)  # (num_qs, B, 1)
        vs = self.shared_value.get_value_single_head(obs_t, head_idx=head_idx)  # (num_qs, B, 1)
        v = vs.mean(dim=0).squeeze(-1)
        q_mean = qs.mean(dim=0).squeeze(-1)
        q_std = qs.std(dim=0).squeeze(-1)
        return q_mean, q_std, v

    @torch.no_grad()
    def get_training_action_with_index(self, raw_obs: np.ndarray, index: int):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
        if index == 0:
            final_action, normalized_action = self._get_learner_training_action(obs_t_learner, self.task_specific_noise, k=1)
        else:
            # Oracle actor always receives unnormalized observations during interaction
            final_action, normalized_action = self._get_oracle_training_action(obs_t_raw, index - 1, k=1)
        return (
            final_action.squeeze(0).detach().cpu().numpy(),
            normalized_action.squeeze(0).detach().cpu().numpy(),
            int(index),
            None,
        )

    @torch.no_grad()
    def get_learner_training_action(self, raw_obs: np.ndarray):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
        noise = self.get_action_noise(self.global_step, self.total_timesteps, self.min_exploration_noise, self.max_exploration_noise)
        final_action, normalized_action = self._get_learner_training_action(obs_t_learner, noise, k=1)
        return (
            final_action.squeeze(0).detach().cpu().numpy(),
            normalized_action.squeeze(0).detach().cpu().numpy(),
            0,
            None,
        )

    @torch.no_grad()
    def score_action_obs(self, obs_rep: torch.Tensor, actions: torch.Tensor, action_candidates_index: torch.Tensor, if_greedy: bool = False):
        """
        obs_rep: (N, D), replicated observation in *critic space* (normalized if enabled)
        actions: (N, A), normalized action candidates
        action_candidates_index: (N,), which policy proposed each candidate (0..n_oracles)
        returns: (N,) scores
        """
        q_mean, q_std, v = self._get_shared_head_training_critic(obs_rep, actions, head_idx=0)
        V_max = v

        effective_kappa = 1e10 if if_greedy else self.kappa
        other_mask = (action_candidates_index != 0).float()
        adjusted_q = q_mean - effective_kappa * q_std * other_mask

        for i in range(self.n_oracles):
            head_idx = i + 1
            q_mean, q_std, v = self._get_shared_head_training_critic(obs_rep, actions, head_idx=head_idx)
            other_mask = (action_candidates_index != head_idx).float()
            q_orc = q_mean - effective_kappa * q_std * other_mask
            V_max = torch.maximum(V_max, v)
            adjusted_q = torch.maximum(adjusted_q, q_orc)

        return adjusted_q - V_max

    @torch.no_grad()
    def prepare_action_candidates_obs(self, learner_obs_t: torch.Tensor, oracle_obs_t: torch.Tensor, k: int = 3):
        """
        learner_obs_t: (B, D), obs for learner actor (normalized if enabled)
        oracle_obs_t: (B, D), obs for oracle actors (raw / unnormalized)
        Returns:
          - final_action_candidates: ((n_oracles+1)*k*B, A_env)
          - action_candidates:      ((n_oracles+1)*k*B, A_norm) normalized action
          - action_candidates_index:((n_oracles+1)*k*B,) policy index
        """
        cand_norm = []
        cand_final = []
        cand_index = []

        a_learner_final, a_learner_norm = self._get_learner_training_action(learner_obs_t, self.task_specific_noise, k)
        cand_norm.append(self._flatten_action_candidates(a_learner_norm))
        cand_final.append(self._flatten_action_candidates(a_learner_final))
        cand_index.append(torch.zeros((k * learner_obs_t.shape[0],), device=self.device, dtype=torch.long))

        for i in range(self.n_oracles):
            a_orc_final, a_orc_norm = self._get_oracle_training_action(oracle_obs_t, i, k)
            cand_norm.append(self._flatten_action_candidates(a_orc_norm))
            cand_final.append(self._flatten_action_candidates(a_orc_final))
            cand_index.append(torch.full((k * learner_obs_t.shape[0],), i + 1, device=self.device, dtype=torch.long))

        return torch.cat(cand_final, dim=0), torch.cat(cand_norm, dim=0), torch.cat(cand_index, dim=0)

    @torch.no_grad()
    def soft_optimistic_sampling(self, final_action_candidates, candidates, scores, temperature: float = 1.0):
        """
        Softmax over scores/temperature to sample candidate action.
        """
        infos = {}
        scores = scores - scores.max()
        raw_proba = torch.softmax(scores, dim=-1)
        raw_entropy = -torch.sum(raw_proba * torch.log(raw_proba + EPS), dim=-1)
        infos["env/raw_entropy"] = raw_entropy.mean()
        logits = scores / float(temperature)
        action_probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + EPS), dim=-1)
        infos["env/entropy"] = entropy.mean()
        selected_action_index = torch.multinomial(action_probs, 1).squeeze()
        final_selected_action = final_action_candidates[selected_action_index, :]
        selected_action = candidates[selected_action_index, :]
        infos["env/selected_action_index"] = selected_action_index
        return final_selected_action, selected_action, infos

    @torch.no_grad()
    def get_guided_training_action(self, raw_obs: np.ndarray):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw

        final_action_candidates, action_candidates, action_candidates_index = self.prepare_action_candidates_obs(
            obs_t_learner, obs_t_raw, k=self.num_proposals
        )
        n_candidates = int(action_candidates.shape[0])
        obs_rep = obs_t_learner.repeat(n_candidates, 1)
        scores = self.score_action_obs(obs_rep, action_candidates, action_candidates_index, if_greedy=self.baseline_greedy_guided)
        temperature = self.get_temperature(self.global_step, self.total_timesteps)
        final_selected_action, selected_action, sampling_infos = self.soft_optimistic_sampling(
            final_action_candidates, action_candidates, scores, temperature=temperature
        )

        k = int(self.num_proposals)
        policy_index = int(sampling_infos["env/selected_action_index"] // max(k, 1))
        sampling_infos["env/selected_action_index"] = policy_index

        exploration_noise = self.get_action_noise(
            self.global_step, self.total_timesteps, self.min_exploration_noise, self.max_exploration_noise
        )
        selected_action = self.add_noise_to_action(selected_action, exploration_noise, clip_noise=self.clip_noise)
        final_selected_action = selected_action * self.learner_actor.action_scale + self.learner_actor.action_bias

        normalized_action = selected_action.squeeze(0).detach().cpu().numpy()
        final_action = final_selected_action.squeeze(0).detach().cpu().numpy()
        return final_action, normalized_action, policy_index, sampling_infos

    @torch.no_grad()
    def get_inference_action(self, raw_obs: np.ndarray, eval_oracle: bool = False, oracle_name: Optional[str] = None):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
        if eval_oracle:
            if oracle_name is None:
                raise ValueError("oracle_name is required when eval_oracle=True")
            idx = self.oracles_2_index[str(oracle_name)]
            final_action, _ = self._get_oracle_training_action(obs_t_raw, idx, k=1)
        else:
            final_action, _ = self._get_learner_training_action(obs_t_learner, 0.0, k=1)
        return final_action.squeeze(0).detach().cpu().numpy()

    # -----------------------
    # Training loop
    # -----------------------
    def run(self, env, replay_buffer: ReplayBuffer, eval_env=None):
        self._pre_run(env, replay_buffer, eval_env=eval_env)
        _eval_env = eval_env if eval_env is not None else env

        # Evaluate each oracle once at start
        for oracle_name in self.oracles_names:
            self.oracles_actors[oracle_name].eval()
            eval_results = self.evaluate(_eval_env, num_episodes=self.eval_episodes, save_name=f"eval_{oracle_name}", eval_oracle=True, oracle_name=oracle_name)
            save_eval_results_to_csv(self.args["checkpoint_dir"], self.args["run_name"], eval_results)
            if self.use_wandb:
                for k, v in eval_results.items():
                    wandb.log({f"eval/{oracle_name}/{k}": v}, step=self.global_step)
            self.oracles_actors[oracle_name].train()

        env_obs, _ = self.env.reset(seed=self.args["seed"])

        while self.global_step < self.total_timesteps:
            self.global_step += 1
            guided_training_infos = None

            if self.normalize_observations:
                self._update_obs_normalizer(np.asarray(env_obs, dtype=np.float32).reshape(1, -1))

            with torch.inference_mode():
                if self.global_step < self.total_explore_steps:
                    actor_cycle_step = (self.global_step - 1) // self.per_oracle_explore_steps
                    actor_index = int(actor_cycle_step % (self.n_oracles + 1))
                    final_action, normalized_action, selected_action_index, guided_training_infos = self.get_training_action_with_index(
                        env_obs, actor_index
                    )
                else:
                    final_action, normalized_action, selected_action_index, guided_training_infos = self.get_guided_training_action(env_obs)

            selected_action_index = int(selected_action_index)
            self.action_selection_buffer.append(selected_action_index)

            next_env_obs, reward, terminated, truncated, infos = self.env.step(final_action)

            if self.normalize_rewards:
                self._update_reward_normalizer(float(reward), bool(terminated or truncated))

            if self.use_wandb and isinstance(infos, dict) and "episode" in infos:
                wandb.log(
                    {"env/episode_reward": infos["episode"]["r"], "env/episode_length": infos["episode"]["l"]},
                    step=self.global_step,
                )

            self.replay_buffer.add_transition(
                {
                    "observations": env_obs,
                    "next_observations": next_env_obs,
                    "actions": normalized_action,
                    "rewards": float(reward),
                    "dones": bool(terminated),
                    "truncated": bool(truncated),
                    "action_mask": int(selected_action_index),
                    "global_step": int(self.global_step),
                }
            )
            env_obs = next_env_obs

            train_info = None
            if self.global_step >= self.learner_learning_starts:
                train_info = self.update()

            do_log = self.use_wandb and (self.global_step % self.log_every == 0)
            if do_log:
                info_payload = train_info or {}
                info_payload["time/total_timesteps"] = self.global_step
                if self.global_step % 500 == 0 and len(self.action_selection_buffer) > 5000: # hard code
                    arr = np.array(self.action_selection_buffer, dtype=int)
                    info_payload["env/selected_index_distribution"] = wandb.Histogram(arr)
                    info_payload["env/learner_selected"] = float((arr == 0).mean())
                    for i in range(self.n_oracles):
                        info_payload[f"env/oracle_{i}_selected"] = float((arr == (i + 1)).mean())
                    info_payload["env/temperature"] = self.get_temperature(self.global_step, self.total_timesteps)
                info_payload["time/fps"] = self.global_step / (time.time() - self.start_time)
                wandb.log(info_payload, step=self.global_step)
                if guided_training_infos is not None:
                    wandb.log(guided_training_infos, step=self.global_step)

            self.progress_bar.update(1)

            if (self.global_step + 1) % self.eval_every == 0:
                self.learner_actor.eval()
                eval_results = self.evaluate(_eval_env, num_episodes=self.eval_episodes)
                self.learner_actor.train()
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
    # Updates
    # -----------------------
    def update(self):
        self.learner_actor.train()
        self.learner_critic.train()
        self.learner_value.train()
        self.shared_critic.train()
        self.shared_value.train()

        num_blocks = int(self.args["learner_critic_utd_ratio"])

        actor_data = self._prepare_sampled_batch(self.replay_buffer.sample(int(self.args["learner_critic_batch_size"])))
        actor_mini_batches = reshape_into_blocks(actor_data, num_blocks)

        learner_data = self._prepare_sampled_batch(
            self.replay_buffer.action_mask_sample(int(self.args["learner_critic_batch_size"]), 0, on_policy_ratio=0.5)
        )
        learner_mini_batches = reshape_into_blocks(learner_data, num_blocks)

        shared_oracle_mini_batches = {}
        for oracle_idx, name in enumerate(self.oracles_names):
            odata = self._prepare_sampled_batch(
                    self.replay_buffer.action_mask_sample(
                        int(self.args["learner_critic_batch_size"]), oracle_idx + 1, on_policy_ratio=0.5
                    )
                )
            shared_oracle_mini_batches[name] = reshape_into_blocks(odata, num_blocks)

        info = {}
        for i in range(num_blocks):
            self.update_step += 1
            learner_sub_batch = {k: v[i] for k, v in learner_mini_batches.items()}

            shared_oracle_sub_batches = {name: {k: v[i] for k, v in mb.items()} for name, mb in shared_oracle_mini_batches.items()}

            critic_info = self.update_critic(learner_sub_batch, shared_oracle_sub_batches)
            info.update(critic_info)
            self.update_target_learner_critic()
            self.update_target_shared()

            if self.update_step % self.policy_delay == 0:
                actor_sub_batch = {k: v[i] for k, v in actor_mini_batches.items()}
                actor_info = self.update_learner_actor(actor_sub_batch)
                info.update(actor_info)
                self.update_target_learner_actor()

        return info

    def update_critic(self, learner_data: dict, shared_oracle_data_by_name: Optional[Dict[str, dict]] = None):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        q_loss, v_loss, info = self._compute_critic_loss(learner_data, shared_oracle_data_by_name)

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        self.shared_critic_optimizer.zero_grad(set_to_none=True)
        self.shared_value_optimizer.zero_grad(set_to_none=True)

        total_loss = q_loss + v_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.learner_critic.parameters(), self.args["max_grad_norm"])
        nn.utils.clip_grad_norm_(self.learner_value.parameters(), self.args["max_grad_norm"])
        nn.utils.clip_grad_norm_(self.shared_critic.parameters(), self.args["max_grad_norm"])
        nn.utils.clip_grad_norm_(self.shared_value.parameters(), self.args["max_grad_norm"])

        self.critic_optimizer.step()
        self.value_optimizer.step()
        l2normalize_network(self.learner_critic)
        l2normalize_network(self.learner_value)

        self.shared_critic_optimizer.step()
        self.shared_value_optimizer.step()
        l2normalize_network(self.shared_critic)
        l2normalize_network(self.shared_value)

        return info

    def _compute_critic_loss(
        self, learner_data: dict, shared_oracle_data_by_name: Optional[Dict[str, dict]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        info: Dict[str, Any] = {}
        total_q_loss = 0.0
        total_v_loss = 0.0

        # Separate learner critic/value (training)
        q_loss, v_loss, q_mean, v_mean, q_uncertainty = self._compiled_compute_learner_critic_loss(learner_data)
        total_q_loss = total_q_loss + q_loss
        total_v_loss = total_v_loss + v_loss
        info.update(
            {
                "separate/learner/qf_loss": q_loss.item(),
                "separate/learner/vf_loss": v_loss.item(),
                "separate/learner/qf_mean": q_mean.item(),
                "separate/learner/vf_mean": v_mean.item(),
                "separate/learner/qf_uncertainty": q_uncertainty.item(),
            }
        )

        # Shared-head critic/value (optional training)
            # Always train learner head (0) on learner batch
        sq, sv, sqm, svm, squ = self._compiled_compute_shared_head_critic_loss(learner_data, 0)
        total_q_loss = total_q_loss + sq
        total_v_loss = total_v_loss + sv
        info.update(
            {
                "learner/qf_loss": sq.item(),
                "learner/vf_loss": sv.item(),
                "learner/qf_mean": sqm.item(),
                "learner/vf_mean": svm.item(),
                "learner/qf_uncertainty": squ.item(),
            })

        for oracle_idx, name in enumerate(self.oracles_names):
            head_idx = oracle_idx + 1
            sq, sv, sqm, svm, squ = self._compiled_compute_shared_head_critic_loss(shared_oracle_data_by_name[name], head_idx)
            total_q_loss = total_q_loss + sq
            total_v_loss = total_v_loss + sv
            info.update({f"oracles/{name}/qf_loss": sq.item(), 
                         f"oracles/{name}/vf_loss": sv.item(), 
                         f"oracles/{name}/qf_mean": sqm.item(), 
                         f"oracles/{name}/vf_mean": svm.item(), 
                         f"oracles/{name}/qf_uncertainty": squ.item()})

        return total_q_loss, total_v_loss, info

    def _compute_shared_head_critic_loss(self, data: dict, head_idx: int):
        """
        Shared-head categorical TD loss for critic and MSE value regression.
        Uses head-specific policy for target action generation.
        """
        head_idx = int(head_idx)

        with torch.no_grad():
            if head_idx == 0:
                next_actions = self.learner_actor.get_target_action(data["next_observations"], self.target_policy_noise)
            else:
                oracle_name = self._oracle_name_from_head_idx(head_idx)
                next_actions = self.oracles_actors[oracle_name].get_action(data["next_observations"], self.oracle_std_multiplier)

            next_qs, next_q_infos = self.shared_critic.get_target_value_with_info_single_head(
                data["next_observations"], next_actions, head_idx=head_idx
            )  # (num_qs, B, 1), infos list len num_qs
            min_idx = next_qs.squeeze(-1).argmin(dim=0)  # (B,)
            stacked_log_probs = torch.stack([info["log_prob"] for info in next_q_infos], dim=0)  # (num_qs, B, num_bins)
            next_q_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)]

        pred_qs, pred_q_infos = self.shared_critic.get_value_with_info_single_head(
            data["observations"], data["actions"], head_idx=head_idx
        )
        gamma_n = float(self.gamma) ** int(self.args.get("nstep", 1))

        q_loss = 0.0
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
            )
            q_loss = q_loss + loss

        q_mean = pred_qs.mean()
        q_uncertainty = pred_qs.std(dim=0).mean()

        with torch.no_grad():
            if head_idx == 0:
                actions = self.learner_actor.get_target_action(data["observations"], self.target_policy_noise)
            else:
                oracle_name = self._oracle_name_from_head_idx(head_idx)
                actions = self.oracles_actors[oracle_name].get_action(data["observations"], self.oracle_std_multiplier)

            critic_values = self.shared_critic.get_target_value_single_head(data["observations"], actions, head_idx=head_idx)
            v_target1 = critic_values[0].squeeze(-1)
            v_target2 = critic_values[1].squeeze(-1)

        v_values = self.shared_value.get_value_single_head(data["observations"], head_idx=head_idx)
        v1, v2 = v_values[0].squeeze(-1), v_values[1].squeeze(-1)
        v_loss = F.mse_loss(v1, v_target1, reduction="none").mean() + F.mse_loss(v2, v_target2, reduction="none").mean()

        return q_loss, v_loss, q_mean, critic_values.mean(), q_uncertainty

    def _compute_learner_critic_loss(self, data: dict):
        """Compute categorical TD loss for *separate learner critic*."""
        with torch.no_grad():
            next_actions = self.learner_actor.get_target_action(data["next_observations"], self.target_policy_noise)
            next_qs, next_q_infos = self.learner_critic.get_target_value_with_info(data["next_observations"], next_actions)
            min_idx = next_qs.squeeze(-1).argmin(dim=0)
            stacked_log_probs = torch.stack([info["log_prob"] for info in next_q_infos], dim=0)
            next_q_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)]

        pred_qs, pred_q_infos = self.learner_critic.get_value_with_info(data["observations"], data["actions"])
        gamma_n = float(self.gamma) ** int(self.args.get("nstep", 1))

        q_loss = 0.0
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
            )
            q_loss = q_loss + loss

        q_mean = pred_qs.mean()
        q_uncertainty = pred_qs.std(dim=0).mean()

        with torch.no_grad():
            actions = self.learner_actor.get_target_action(data["observations"], self.target_policy_noise)
            learner_critic_values = self.learner_critic.get_target_value(data["observations"], actions)
            v_target1 = learner_critic_values[0].squeeze(-1)
            v_target2 = learner_critic_values[1].squeeze(-1)

        v_values = self.learner_value.get_value(data["observations"])
        v1, v2 = v_values[0].squeeze(-1), v_values[1].squeeze(-1)
        v_loss = F.mse_loss(v1, v_target1, reduction="none").mean() + F.mse_loss(v2, v_target2, reduction="none").mean()
        return q_loss, v_loss, q_mean, v_values.mean(), q_uncertainty

    def _compute_learner_actor_loss(self, data: dict):
        """
        Actor update:
          - AWBC weights use max-advantage estimate; learner values come from separate learner critic/value.
          - Optional: oracle max comes from shared-head estimators (no gradients).
          - RL loss term uses *separate learner critic* (TD3-like).
        """
        actions = data["actions"]

        with torch.no_grad():
            qs = self.shared_critic.get_value_single_head(data["observations"], actions, head_idx=0)
            q_max = qs.mean(0).squeeze(-1)
            vs = self.shared_value.get_value_single_head(data["observations"], head_idx=0)
            v_max = vs.mean(0).squeeze(-1)

            for oracle_idx in range(self.n_oracles):
                head_idx = oracle_idx + 1
                qk = self.shared_critic.get_value_single_head(data["observations"], actions, head_idx=head_idx).mean(0).squeeze(-1)
                q_max = torch.maximum(q_max, qk)
            for oracle_idx in range(self.n_oracles):
                head_idx = oracle_idx + 1
                vk = self.shared_value.get_value_single_head(data["observations"], head_idx=head_idx).mean(0).squeeze(-1)
                v_max = torch.maximum(v_max, vk)

            advantage = q_max - v_max
            positive_ratio = (advantage > 0).float().mean()
            weights = torch.exp((self.beta * advantage).clamp(-20.0, 50.0))

        predicted_action = self.learner_actor.get_eval_action(data["observations"])
        raw_loss = (predicted_action - data["actions"]).pow(2).sum(dim=-1)
        awbc_loss = (weights.detach() * raw_loss).mean()

        a_pi = self.learner_actor.get_eval_action(data["observations"])
        q_pi = self.learner_critic.get_value(data["observations"], a_pi).mean(0).squeeze(-1)
        td3_obj = (-1.0 * q_pi).mean()

        return awbc_loss, td3_obj, q_pi.mean(), weights.mean(), positive_ratio

    def get_actor_loss_weight(self):
        if self.use_adaptive_actor_loss_weight:
            ratio_of_select_learner = self.global_step / self.total_timesteps
            bc_weight = 1.0 * (1.0 - ratio_of_select_learner)
            rl_weight = ratio_of_select_learner
        else:
            bc_weight = self.l1_loss_weight
            rl_weight = self.rl_loss_weight
        return {"rl_weight": rl_weight, "bc_weight": bc_weight}

    def update_learner_actor(self, data: dict):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        awbc_loss, rl_loss, q_pi_mean, actor_weight, weight_positive_ratio = self._compiled_compute_learner_actor_loss(data)

        self.actor_optimizer.zero_grad(set_to_none=True)
        weights = self.get_actor_loss_weight()
        total_actor_loss = rl_loss * weights["rl_weight"] + awbc_loss * weights["bc_weight"]
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.learner_actor.parameters(), self.args["policy_max_grad_norm"])
        self.actor_optimizer.step()
        l2normalize_network(self.learner_actor)

        return {
            "learner/losses/qf_mean": q_pi_mean.item(),
            "learner/losses/rl_loss": rl_loss.item(),
            "learner/losses/awbc_loss": awbc_loss.item(),
            "learner/losses/rl_weight": weights["rl_weight"],
            "learner/losses/bc_weight": weights["bc_weight"],
            "learner/losses/maxadv_weight": actor_weight.item(),
            "learner/losses/positive_ratio": weight_positive_ratio.item(),
        }

    # -----------------------
    # Target updates
    # -----------------------
    def update_target_shared(self):
        with torch.no_grad():
            self.shared_critic.update_target(self.args["tau"])
            self.shared_value.update_target(self.args["tau"])

    def update_target_learner_critic(self):
        with torch.no_grad():
            self.learner_critic.update_target(self.args["tau"])
            self.learner_value.update_target(self.args["tau"])

    def update_target_learner_actor(self):
        with torch.no_grad():
            self.learner_actor.update_target(self.args["tau"])

    # -----------------------
    # BaseAlgo checkpoint hooks
    # -----------------------
    def _save_model_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint.update(
            {
                "learner_actor_state_dict": self.learner_actor.state_dict(),
                "learner_critic_state_dict": self.learner_critic.state_dict(),
                "learner_value_state_dict": self.learner_value.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "value_optimizer_state_dict": self.value_optimizer.state_dict(),
                "shared_critic_state_dict": self.shared_critic.state_dict(),
                "shared_value_state_dict": self.shared_value.state_dict(),
                "oracles_names": list(self.oracles_names),
            }
        )

        if self.shared_critic_optimizer is not None:
            checkpoint["shared_critic_optimizer_state_dict"] = self.shared_critic_optimizer.state_dict()
        if self.shared_value_optimizer is not None:
            checkpoint["shared_value_optimizer_state_dict"] = self.shared_value_optimizer.state_dict()

        for i, name in enumerate(self.oracles_names):
            checkpoint[f"oracles_actor_state_dict_{i}"] = self.oracles_actors[name].state_dict()

        checkpoint["rng_state_torch"] = torch.get_rng_state()
        checkpoint["rng_state_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        checkpoint["rng_state_numpy"] = np.random.get_state()
        checkpoint["rng_state_python"] = random.getstate()

        buf = np.array(self.action_selection_buffer, dtype=int) if len(self.action_selection_buffer) > 0 else np.array([], dtype=int)
        checkpoint["action_selection_buffer"] = buf
        checkpoint["action_selection_buffer_maxlen"] = int(self.action_selection_buffer.maxlen or self.args.get("action_selection_buffer_size", 0))

        if self.obs_normalizer is not None:
            checkpoint["obs_rms"] = self.obs_normalizer.rms.state_dict()
        if self.reward_normalizer is not None:
            checkpoint["reward_norm"] = {
                "G": self.reward_normalizer.G,
                "G_rms": self.reward_normalizer.G_rms.state_dict(),
                "G_r_max": self.reward_normalizer.G_r_max,
            }

        return checkpoint

    def _load_model_checkpoint(self, checkpoint: dict) -> dict:
        self.learner_actor.load_state_dict(checkpoint["learner_actor_state_dict"])
        self.learner_critic.load_state_dict(checkpoint["learner_critic_state_dict"])
        if "learner_value_state_dict" in checkpoint:
            self.learner_value.load_state_dict(checkpoint["learner_value_state_dict"])

        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        if "value_optimizer_state_dict" in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])

        if "shared_critic_state_dict" in checkpoint:
            self.shared_critic.load_state_dict(checkpoint["shared_critic_state_dict"])
        if "shared_value_state_dict" in checkpoint:
            self.shared_value.load_state_dict(checkpoint["shared_value_state_dict"])

        if self.shared_critic_optimizer is not None and "shared_critic_optimizer_state_dict" in checkpoint:
            self.shared_critic_optimizer.load_state_dict(checkpoint["shared_critic_optimizer_state_dict"])
        if self.shared_value_optimizer is not None and "shared_value_optimizer_state_dict" in checkpoint:
            self.shared_value_optimizer.load_state_dict(checkpoint["shared_value_optimizer_state_dict"])

        for i, name in enumerate(self.oracles_names):
            k_actor = f"oracles_actor_state_dict_{i}"
            if k_actor in checkpoint:
                self.oracles_actors[name].load_state_dict(checkpoint[k_actor])

        try:
            if "rng_state_torch" in checkpoint:
                torch.set_rng_state(checkpoint["rng_state_torch"])
            if "rng_state_cuda" in checkpoint and checkpoint["rng_state_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["rng_state_cuda"])
            if "rng_state_numpy" in checkpoint:
                np.random.set_state(checkpoint["rng_state_numpy"])
            if "rng_state_python" in checkpoint:
                random.setstate(checkpoint["rng_state_python"])
        except Exception as e:
            print(f"[SharedMaxAdv] Warning: failed to restore RNG state: {e}")

        if "action_selection_buffer" in checkpoint:
            buf = checkpoint["action_selection_buffer"]
            maxlen = int(checkpoint.get("action_selection_buffer_maxlen", self.args.get("action_selection_buffer_size", 0)))
            try:
                seq = buf.tolist() if hasattr(buf, "tolist") else list(buf)
            except Exception:
                seq = []
            self.action_selection_buffer = deque(seq, maxlen=maxlen)

        if "obs_rms" in checkpoint and self.obs_normalizer is not None:
            self.obs_normalizer.rms.load_state_dict(checkpoint["obs_rms"])

        if "reward_norm" in checkpoint and self.reward_normalizer is not None:
            self.reward_normalizer.load_state_dict(checkpoint["reward_norm"])
            self.G = self.reward_normalizer.G
            self.G_r_max = self.reward_normalizer.G_r_max

        return checkpoint


# Back-compat alias (old class name)
MaxAdv_Shared = SharedMaxAdv