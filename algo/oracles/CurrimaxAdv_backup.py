import random
import time
import math
from collections import deque
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
torch.set_float32_matmul_precision('high')
from algo.base_algo import BaseAlgo
from algo.algo_utils import ObservationNormalizer, RewardNormalizer, RunningMeanStd, categorical_td_loss, reshape_into_blocks, save_eval_results_to_csv
from data_buffer.replay_buffer import ReplayBuffer
from model.simba import DeterministicSimbaPolicy, SimbaCritics, SimbaValues
from model.simba_base import EPS, l2normalize_network
from model.mlp import OraclePolicyBase



class CurrMaxAdv(BaseAlgo):
    def __init__(
        self,
        n_oracles: int,
        oracles_actors: Dict[str, OraclePolicyBase],
        oracles_critics: Dict[str, SimbaCritics],
        oracles_values: Dict[str, SimbaValues],
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
        self.oracles_critics = dict(oracles_critics)
        self.oracles_values = dict(oracles_values)

        self.oracles_names = list(self.oracles_actors.keys())
        self.oracles_2_index = {name: i for i, name in enumerate(self.oracles_names)}
        assert len(self.oracles_names) == len(self.oracles_critics) == len(self.oracles_values) == self.n_oracles

        # Move models to device + build list views for hot path
        for name in self.oracles_names:
            self.oracles_actors[name] = self.oracles_actors[name].to(device)
            self.oracles_critics[name] = self.oracles_critics[name].to(device)
            self.oracles_values[name] = self.oracles_values[name].to(device)
        self._oracle_actor_list = [self.oracles_actors[name] for name in self.oracles_names]
        self._oracle_critic_list = [self.oracles_critics[name] for name in self.oracles_names]
        self._oracle_value_list = [self.oracles_values[name] for name in self.oracles_names]

        self.learner_actor = learner_actor.to(device)
        self.learner_critic = learner_critic.to(device)
        self.learner_value = learner_value.to(device)

        # -----------------------
        # Hyperparameters
        # -----------------------
        # Oracle std multipliers: allow scalar or per-oracle list
        _osm = self.args["oracle_std_multiplier"]
        if isinstance(_osm, (list, tuple, np.ndarray)):
            self.oracle_std_multipliers = [float(x) for x in list(_osm)]
            assert len(self.oracle_std_multipliers) == self.n_oracles, (
                f"oracle_std_multiplier must have length n_oracles={self.n_oracles}, "
                f"got {len(self.oracle_std_multipliers)}"
            )
        else:
            self.oracle_std_multipliers = [float(_osm) for _ in range(self.n_oracles)]
        self.oracle_std_multiplier_by_name = {
            name: self.oracle_std_multipliers[i] for i, name in enumerate(self.oracles_names)
        }
        # Back-compat: keep scalar field (use first oracle's multiplier)
        self.oracle_std_multiplier = float(self.oracle_std_multipliers[0]) if self.n_oracles > 0 else float(_osm)

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
        self.replay_ratio = float(self.args["replay_ratio"])
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
        self.normalized_g_max = float(self.args.get("normalized_g_max", 5.0))

        # Oracle warmup schedule
        self.per_oracle_explore_steps = int(self.args["per_oracle_explore_steps"])
        self.total_explore_steps = int(self.per_oracle_explore_steps * (self.n_oracles + 1))

        # -----------------------
        # Curriculum: consecutive actor commitment
        # -----------------------
        # When >1, after we sample a policy index we keep using that same index for
        # `consecutive_steps` env steps, then resample.
        #
        # Schedule: every `total_consecutive_steps` env steps we linearly decrease the
        # commitment length until it reaches `min_consecutive_steps` (>=1).
        #
        # Defaults keep original behavior (select policy every step).
        self.max_consecutive_steps = int(self.args["max_consecutive_steps"])
        self.min_consecutive_steps = int(self.args["min_consecutive_steps"])
        self.max_consecutive_steps = max(1, self.max_consecutive_steps)
        self.min_consecutive_steps = max(1, self.min_consecutive_steps)
        self.total_consecutive_steps = max(1, self.args["total_consecutive_steps"])

        # Internal state for commitment
        self._committed_policy_index: Optional[int] = None
        self._committed_steps_left: int = 0

        # Logging / controls
        self.log_every = int(self.args.get("log_every", 10))
        self.action_selection_buffer = deque(maxlen=int(self.args["action_selection_buffer_size"]))

        # Baselines / toggles
        self.baseline_greedy_guided = bool(self.args.get("baseline_greedy_guided", False))
        self.scheduled_exploration = bool(self.args.get("scheduled_exploration", False)) 
        self.scheduled_exploration_steps = int(self.args.get("scheduled_exploration_steps", 100000)) 
        self.scheduled_exploration_prob = float(self.args.get("scheduled_exploration_prob", 0.5)) 

        # Mixed precision / compilation
        self.use_compile = bool(self.args.get("use_compile", False))
        # Compiling rollout-time helpers (action sampling / critic scoring) is often counterproductive:
        # those paths take Python scalars (noise_std, k) and do Python-side oracle indexing, which can
        # trigger frequent recompilations and hit TorchDynamo's small default cache limit. Keep it opt-in.
        self.compile_env_interaction = bool(self.args.get("compile_env_interaction", True))

        # -----------------------
        # Normalizers
        # -----------------------
        self.obs_normalizer = None
        self.obs_rms = None
        if self.normalize_observations:
            # Models expose obs_dim
            obs_dim = int(self.learner_actor.obs_dim)
            self.obs_normalizer = ObservationNormalizer(obs_dim=obs_dim, dtype=np.float32, eps=float(EPS))
            # Back-compat: checkpoint payload key expects `obs_rms`
            self.obs_rms = self.obs_normalizer.rms

        self.reward_normalizer = None
        if self.normalize_rewards:
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.gamma), normalized_g_max=float(self.normalized_g_max), dtype=np.float32, eps=float(EPS)
            )
            # Back-compat fields
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

        self.oracles_critic_optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.oracles_value_optimizers: Dict[str, torch.optim.Optimizer] = {}
        for name in self.oracles_names:
            self.oracles_critic_optimizers[name] = self.oracles_critics[name].set_optimizer(self.args["critic_lr"], weight_decay=critic_wd)
            self.oracles_value_optimizers[name] = self.oracles_values[name].set_optimizer(self.args["value_lr"], weight_decay=value_wd)

        # -----------------------
        # Compilation hooks
        # -----------------------
        self._compiled_compute_oracles_critic_loss = self._compute_oracles_critic_loss
        self._compiled_compute_learner_critic_loss = self._compute_learner_critic_loss
        self._compiled_compute_learner_actor_loss = self._compute_learner_actor_loss

        if self.use_compile and hasattr(torch, "compile"):
            self._compiled_compute_oracles_critic_loss = self._try_compile(
                self._compiled_compute_oracles_critic_loss, "oracles_critic_loss", fullgraph=True
            )
            self._compiled_compute_learner_critic_loss = self._try_compile(
                self._compiled_compute_learner_critic_loss, "learner_critic_loss", fullgraph=True
            )
            self._compiled_compute_learner_actor_loss = self._try_compile(
                self._compiled_compute_learner_actor_loss, "learner_actor_loss", fullgraph=True
            )

            # Optional: compile env-interaction helpers (rollout-time). Off by default for stability.
            if self.compile_env_interaction:
                # Note: action sampling uses torch.distributions and can trigger TorchDynamo "torch.Size" unsupported
                # when fullgraph=True (see stack trace through Distribution._extended_shape()).
                # We allow graph breaks here by compiling with fullgraph=False.
                self._get_learner_training_action = self._try_compile(
                    self._get_learner_training_action, "get_learner_training_action", fullgraph=False
                )
                self._get_oracle_training_action = self._try_compile(
                    self._get_oracle_training_action, "get_oracle_training_action", fullgraph=False
                )
                self._get_learner_training_critic = self._try_compile(
                    self._get_learner_training_critic, "get_learner_training_critic", fullgraph=True
                )
                self._get_oracle_training_critic = self._try_compile(
                    self._get_oracle_training_critic, "get_oracle_training_critic", fullgraph=True
                )
                self.prepare_action_candidates_obs = self._try_compile(
                    self.prepare_action_candidates_obs, "prepare_action_candidates_obs", fullgraph=False
                )
        print(
            f"MaxAdv: compile={'on' if self.use_compile else 'off'} "
            f"(env_interaction={'on' if (self.use_compile and self.compile_env_interaction) else 'off'})"
        )

    # -----------------------
    # torch.compile helper
    # -----------------------
    def _try_compile(self, fn, name: str, *, fullgraph: bool = True):
        try:
            # Avoid crashing on models with a few legitimate recompiles (or varying call signatures).
            # Default cache_size_limit is small (8) and can be exceeded easily.
            try:
                import torch._dynamo as dynamo  # type: ignore

                limit = int(self.args.get("dynamo_cache_size_limit", 64))
                if limit > 0:
                    dynamo.config.cache_size_limit = max(int(dynamo.config.cache_size_limit), limit)
            except Exception:
                pass
            return torch.compile(fn, fullgraph=fullgraph)
        except Exception as e:
            print(f"[MaxAdv] torch.compile failed for {name}: {e}. Falling back to eager.")
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
        """Add clipped Gaussian noise to action tensor."""
        eps = torch.randn_like(action)
        eps.mul_(noise)
        eps.clamp_(-clip_noise, clip_noise)
        action = action.add(eps)
        return action.clamp_(-1.0, 1.0)
      
    
    def get_temperature(self, current_step: int, max_steps: int) -> float:
        """Get temperature for stochastic sampling with linear decay"""
        if current_step >= max_steps:
            return float(self.sampling_min_temperature)
        progress = float(current_step) / float(max_steps)
        return float(self.sampling_max_temperature + (self.sampling_min_temperature - self.sampling_max_temperature) * progress)

    # -----------------------
    # Curriculum schedule helpers
    # -----------------------
    def get_scheduled_exploration_prob(self, global_step: int) -> float:
        """
        Scheduled exploration probability schedule.

        If enabled, returns a probability that linearly decays from
        `scheduled_exploration_prob` at step 0 to 0 at `scheduled_exploration_steps`.
        """
        if not bool(self.scheduled_exploration):
            return 0.0

        max_steps = max(1, int(self.scheduled_exploration_steps))
        # Use 0-based step for schedule so step=1 is still ~near initial prob.
        t = max(0, int(global_step) - 1)
        if t >= max_steps:
            return 0.0

        p0 = float(self.scheduled_exploration_prob)
        progress = float(t) / float(max_steps)
        p = p0 * (1.0 - progress)
        # Clamp for safety.
        return float(max(0.0, min(1.0, p)))

    def get_consecutive_steps(self, global_step: int) -> int:
        """
        Commitment length schedule.

        Linearly decrease from `max_consecutive_steps` to `min_consecutive_steps`
        over `total_consecutive_steps` environment steps.
        """
        max_k = int(self.max_consecutive_steps)
        min_k = int(self.min_consecutive_steps)
        if max_k <= min_k:
            return min_k
        
        # Linear decay from max_k to min_k
        decay_steps = int(self.total_consecutive_steps)
        if global_step >= decay_steps:
            return min_k
        
        progress = float(global_step) / float(decay_steps)
        k = float(max_k) + (float(min_k) - float(max_k)) * progress
        return int(max(min_k, min(max_k, round(k))))

    @torch.no_grad()
    def get_committed_training_action(self, raw_obs: np.ndarray, policy_index: int):
        """
        Get an env action for a fixed policy index (no scoring/sampling across policies).

        We mirror the action noise handling from `get_guided_training_action`:
        - compute a normalized action from the selected policy (learner/oracle)
        - add exploration noise (decaying schedule)
        - scale to env action space
        """
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw

        idx = int(policy_index)
        if idx <= 0:
            # Learner policy (0)
            # Positional args keep torch.compile call signatures stable when enabled.
            _final, normalized_action = self._get_learner_training_action(obs_t_learner, self.task_specific_noise, 1)
        else:
            # Oracle policies (1..n_oracles)
            _final, normalized_action = self._get_oracle_training_action(obs_t_raw, idx - 1, 1)

        exploration_noise = self.get_action_noise(
            self.global_step, self.total_timesteps, self.min_exploration_noise, self.max_exploration_noise
        )
        selected_action = self.add_noise_to_action(normalized_action, exploration_noise, clip_noise=self.clip_noise)
        final_selected_action = selected_action * self.learner_actor.action_scale + self.learner_actor.action_bias
        return (
            final_selected_action.squeeze(0).detach().cpu().numpy(),
            selected_action.squeeze(0).detach().cpu().numpy(),
        )
    
    
    # -----------------------
    # Normalization (SIMBA-style)
    # -----------------------
    def _normalize_obs_np(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_normalizer is None:
            return obs
        return self.obs_normalizer.normalize_np(obs)

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
            # Preserve raw tensors so we can feed *raw* observations to components that
            # own their own observation normalizer (e.g., Simba oracle wrappers).
            # NOTE: we keep these as separate keys to avoid accidental double-normalization.
            if "observations_raw" not in batch:
                batch["observations_raw"] = batch["observations"]
            if "next_observations_raw" not in batch:
                batch["next_observations_raw"] = batch["next_observations"]
            batch["observations"] = self._normalize_obs_tensor(batch["observations"])
            batch["next_observations"] = self._normalize_obs_tensor(batch["next_observations"])
        if self.normalize_rewards:
            batch["rewards"] = self._scale_reward_tensor(batch["rewards"])
        return batch

    # -----------------------
    # Tensor helpers
    # -----------------------
    def _obs_to_tensor(self, raw_obs: np.ndarray) -> torch.Tensor:
        """
        Convert raw env observation to a tensor on device (no normalization).
        Note: we normalize observations for *training batches* in `_prepare_sampled_batch`.
        For online action selection, we only normalize inputs to the learner actor (see below).
        """
        obs_np = np.asarray(raw_obs, dtype=np.float32)
        return torch.from_numpy(obs_np).to(self.device).unsqueeze(0)  # (1, D)

    @staticmethod
    def _flatten_action_candidates(actions: torch.Tensor) -> torch.Tensor:
        """
        Flatten sampled actions from shape (k, B, A) to (k*B, A).
        """
        if actions.dim() == 2:
            # (B, A)
            return actions
        if actions.dim() == 3:
            k, b, a = actions.shape
            return actions.reshape(k * b, a)
    # -----------------------
    # Interaction (policy + critics)
    # -----------------------
    @torch.no_grad()
    def _get_learner_training_action(self, obs_t: torch.Tensor, noise_std: float, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_action = self.learner_actor.get_action(obs_t, noise_std) if k == 1 else self.learner_actor.sample_action(obs_t, noise_std, k)
        final_action = normalized_action * self.learner_actor.action_scale + self.learner_actor.action_bias
        return final_action, normalized_action

    @torch.no_grad()
    def _get_oracle_training_action(self, obs_t: torch.Tensor, oracle_index: int, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        oracle_name = self.oracles_names[int(oracle_index)]
        actor = self.oracles_actors[oracle_name]
        std = float(self.oracle_std_multipliers[int(oracle_index)])
        normalized_action = actor.get_action(obs_t, std) if k == 1 else actor.sample_action(obs_t, std, k)
        final_action = normalized_action * actor.action_scale + actor.action_bias
        return final_action, normalized_action

    @torch.no_grad()
    def _get_learner_training_critic(self, obs_t: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qs = self.learner_critic.get_value(obs_t, actions)  # (num_qs, B, 1)
        vs = self.learner_value.get_value(obs_t)  # (num_qs, B, 1)
        v = vs.mean(dim=0).squeeze(-1)
        q_mean = qs.mean(dim=0).squeeze(-1)
        q_std = qs.std(dim=0).squeeze(-1)
        return q_mean, q_std, v

    @torch.no_grad()
    def _get_oracle_training_critic(self, obs_t: torch.Tensor, actions: torch.Tensor, oracle_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        oracle_name = self.oracles_names[int(oracle_index)]
        qs = self.oracles_critics[oracle_name].get_value(obs_t, actions)  # (num_qs, B, 1)
        q_mean = qs.mean(dim=0).squeeze(-1)
        q_std = qs.std(dim=0).squeeze(-1)
        return q_mean, q_std
    
    
    @torch.no_grad()
    def get_training_action_with_index(self,raw_obs,index:int):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
        if index == 0:
            noise = self.task_specific_noise
            final_action, normalized_action = self._get_learner_training_action(obs_t_learner, noise, 1)
        else:
            # IMPORTANT: oracle actor always receives unnormalized observations
            final_action, normalized_action = self._get_oracle_training_action(obs_t_raw, index - 1, 1)
        return (
            final_action.squeeze(0).detach().cpu().numpy(),
            normalized_action.squeeze(0).detach().cpu().numpy(),
            index,
            None,)
    
    ### BASELINE: PURE OFFLINE ########
    @torch.no_grad()
    def get_learner_training_action(self,raw_obs):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
        noise = self.get_action_noise(self.global_step, self.total_timesteps, self.min_exploration_noise, self.max_exploration_noise)
        final_action, normalized_action = self._get_learner_training_action(obs_t_learner, noise, 1)
        return (
            final_action.squeeze(0).detach().cpu().numpy(),
            normalized_action.squeeze(0).detach().cpu().numpy(),
            0,
            None,
        )
    ### BASELINE: PURE OFFLINE ########
    
    @torch.no_grad()
    def score_action_obs(self, obs_rep: torch.Tensor, actions: torch.Tensor, action_candidates_index: torch.Tensor, if_greedy: bool = False):
        """
        obs_rep: nested dict with batch size N (replicated observation)
        actions: (N, A), action candidates proposed by the oralces and learner
        if_greedy: if True, use Q value directly for own actions and -inf for others; 
                   if False, use UCB/LCB adjustments
        return: (N,), maxQ - maxV value of the actions based on the UCB of oracles and LCB of the learner, the shape is (N,)
                and (N,), the critic index (0=learner, 1..n_oracles) that produced the max score
        """
        # ---- compute V_max(s) across all policies ----
        q_mean, q_std, v = self._get_learner_training_critic(obs_rep, actions)  # (N,)
        V_max = v 
        
        # ---- compute adjusted Q (UCB/LCB) for each policy ----
        # Use large penalty instead of -inf for compile-friendliness (avoids torch.where and tensor creation)
        # 1e10 is effectively -inf for Q value comparisons
        effective_kappa = 1e10 if if_greedy else self.kappa
        
        other_mask = (action_candidates_index != 0).float()  # (N,) 1.0 for non-learner actions
        adjusted_q = q_mean - effective_kappa * q_std * other_mask  # (N,)
        best_critic_index = torch.zeros_like(action_candidates_index)
        
        for i in range(self.n_oracles):
            q_mean, q_std = self._get_oracle_training_critic(obs_rep, actions, i)  # (N,)
            other_mask = (action_candidates_index != (i + 1)).float()  # (N,) 1.0 for non-oracle-i actions
            q_orc = q_mean - effective_kappa * q_std * other_mask
            better_mask = (q_orc > adjusted_q).long()
            adjusted_q = torch.maximum(adjusted_q, q_orc)
            # Avoid torch.where for compile-friendliness
            best_critic_index = best_critic_index + (i + 1 - best_critic_index) * better_mask
        
        scores = adjusted_q # (N,)
        return scores, best_critic_index
    
    
    @torch.no_grad()
    def prepare_action_candidates_obs(self, learner_obs_t: torch.Tensor, oracle_obs_t: torch.Tensor, k: int = 3):
        """
        learner_obs_t: observation tensor (B, D) for learner (possibly normalized)
        oracle_obs_t: observation tensor (B, D) for oracles (raw / unnormalized)
        k: (int), the number of action candidates to propose for each policy
        return: (N, A), action candidates proposed by the oralces and learner, the shape is ((N_oracles + 1) * k * B, A) if B>1
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
    def soft_optimistic_sampling(self,final_action_candidates,candidates, scores, temperature = 1.0):
        """
        candidates: (N, A), action candidates proposed by the oralces and learner, the shape is ((N_oracles + 1) * k, A)
        temperature: (float), the temperature for the softmax function
        return: single action to execute, and other info to log 
        
        weight: proportional to exp(score/temperature) / normalizing factor * behavior prior ** eta, prevent very OOD actions
        """
        infos = {}
        ## Normalize the scores, softmax is invariant to translation, but sensitive to scaling, we here substract max for numerical stability
        scores = scores - scores.max()
        ## Normalize end 
        
        raw_proba =torch.softmax(scores, dim=-1)
        raw_entropy = -torch.sum(raw_proba * torch.log(raw_proba + EPS), dim=-1)
        infos["env/raw_entropy"] = raw_entropy.mean()
        logits = scores / temperature # (N,)
        logits = logits.reshape(-1)
        action_probs = torch.softmax(logits, dim=-1)
        action_probs = action_probs.reshape(-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + EPS), dim=-1)
        infos["env/entropy"] = entropy.mean()
        selected_action_index = torch.multinomial(action_probs, 1).squeeze()
        final_selected_action = final_action_candidates[selected_action_index, :]
        selected_action = candidates[selected_action_index, :]
        infos["env/selected_action_index"] = selected_action_index
        return final_selected_action, selected_action, infos
        
        

    @torch.no_grad()
    def get_guided_training_action(self,raw_obs):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
        final_action_candidates, action_candidates, action_candidates_index = self.prepare_action_candidates_obs(
            obs_t_learner, obs_t_raw, self.num_proposals
        )
        n_candidates = action_candidates.shape[0]
        # critics are trained/queried on normalized observations when enabled
        obs_rep = obs_t_learner.repeat(n_candidates, 1)
        scores, best_critic_index = self.score_action_obs(
            obs_rep,
            action_candidates,
            action_candidates_index,
            if_greedy=self.baseline_greedy_guided,
        )
        temperature = self.get_temperature(self.global_step,self.total_timesteps)
        final_selected_action, selected_action, sampling_infos = self.soft_optimistic_sampling(final_action_candidates,action_candidates,scores,temperature=temperature)
        k = int(self.num_proposals)
        selected_action_index = sampling_infos["env/selected_action_index"]
        policy_index = selected_action_index // max(k, 1)
        sampling_infos["env/selected_action_index"] = best_critic_index[selected_action_index]
        
        exploration_noise = self.get_action_noise(self.global_step,self.total_timesteps,self.min_exploration_noise,self.max_exploration_noise)
        selected_action = self.add_noise_to_action(selected_action,exploration_noise,clip_noise=self.clip_noise)
        final_selected_action = selected_action * self.learner_actor.action_scale + self.learner_actor.action_bias
        
        normalized_action = selected_action.squeeze(0).detach().cpu().numpy()
        final_action = final_selected_action.squeeze(0).detach().cpu().numpy()
        # return final_action, normalized_action, policy_index, sampling_infos
        return final_action, normalized_action, sampling_infos["env/selected_action_index"], sampling_infos
    
    
    @torch.no_grad()
    def get_inference_action(self,raw_obs,eval_oracle = False,oracle_name = None):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
        if eval_oracle:
            idx = self.oracles_2_index[str(oracle_name)]
            final_action, _ = self._get_oracle_training_action(obs_t_raw, idx, 1)
        else:
            final_action, _ = self._get_learner_training_action(obs_t_learner, 0.0, 1)
        return final_action.squeeze(0).detach().cpu().numpy()
    

    def run(self, env, replay_buffer: ReplayBuffer, eval_env=None):
        """
        Main training loop. Delegates resume/wandb/progress/finalize to BaseAlgo.
        """
        self._pre_run(env, replay_buffer, eval_env=eval_env)
        _eval_env = eval_env if eval_env is not None else env

        # # Evaluate each oracle once at start (kept from original logic)
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

            # Update observation stats on raw obs
            if self.normalize_observations:
                self._update_obs_normalizer(np.asarray(env_obs, dtype=np.float32).reshape(1, -1))

            with torch.inference_mode():
                if self.global_step < self.total_explore_steps:
                    actor_cycle_step = (self.global_step - 1) // self.per_oracle_explore_steps
                    actor_index = int(actor_cycle_step % (self.n_oracles + 1))
                    final_action, normalized_action, selected_action_index, guided_training_infos = self.get_training_action_with_index(env_obs, actor_index)
                else:
                    # Curriculum: commit to the sampled policy for multiple env steps.
                    consecutive_steps = self.get_consecutive_steps(self.global_step)
                    need_resample = (self._committed_policy_index is None) or (self._committed_steps_left <= 0)
                    if need_resample:
                        # Scheduled exploration baseline: with decayed prob, force a random oracle
                        # (instead of guided sampling) to interact with the environment.
                        sched_p = self.get_scheduled_exploration_prob(self.global_step)
                        do_sched = (sched_p > 0.0) and (random.random() < sched_p)
                        if do_sched and self.n_oracles > 0:
                            chosen_oracle_index = random.randint(1, int(self.n_oracles))  # 1..n_oracles
                            final_action, normalized_action, selected_action_index, guided_training_infos = self.get_training_action_with_index(
                                env_obs, chosen_oracle_index
                            )
                            if guided_training_infos is None:
                                guided_training_infos = {}
                            guided_training_infos["env/scheduled_exploration"] = 1.0
                            guided_training_infos["env/scheduled_exploration_prob"] = float(sched_p)
                            guided_training_infos["env/scheduled_exploration_selected_index"] = float(chosen_oracle_index)
                        else:
                            final_action, normalized_action, selected_action_index, guided_training_infos = self.get_guided_training_action(env_obs)
                            if guided_training_infos is not None:
                                guided_training_infos["env/scheduled_exploration"] = 0.0
                                guided_training_infos["env/scheduled_exploration_prob"] = float(sched_p)
                        self._committed_policy_index = int(selected_action_index)
                        # We've already consumed 1 step with this selection.
                        self._committed_steps_left = int(consecutive_steps) - 1
                        if guided_training_infos is None:
                            guided_training_infos = {}
                        guided_training_infos["env/consecutive_steps"] = float(consecutive_steps)
                        guided_training_infos["env/consecutive_steps_left"] = float(self._committed_steps_left)
                        guided_training_infos["env/committed_policy_index"] = float(self._committed_policy_index)
                    else:
                        final_action, normalized_action = self.get_committed_training_action(env_obs, int(self._committed_policy_index))
                        selected_action_index = int(self._committed_policy_index)
                        self._committed_steps_left -= 1
                    # print(f"Committed policy index: {self._committed_policy_index}, consecutive steps left: {self._committed_steps_left}")

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

            # Store transition (numpy obs)
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
                # Reset commitment at episode boundaries for cleaner curriculum behavior.
                self._committed_policy_index = None
                self._committed_steps_left = 0

        return self._after_run()
    
    
    def update(self):
        """We use different utd ratio for critic and actor.

        Critic/value updates:
          - Learner and each oracle use their own *mixed* batch (own transitions + global mixed).
        Actor updates:
          - Learner actor uses the standard mixed replay batch (no action_mask filtering).
        """
        self.learner_actor.train()
        self.learner_critic.train()
        self.learner_value.train()
        for name in self.oracles_names:
            self.oracles_actors[name].train()
            self.oracles_critics[name].train()
            self.oracles_values[name].train()

        num_blocks = int(self.args["learner_critic_utd_ratio"])

        # --- Actor dataset: standard mixed replay ---
        actor_data = self._prepare_sampled_batch(
            self.replay_buffer.sample(int(self.args["learner_critic_batch_size"]))
        )
        actor_mini_batches = reshape_into_blocks(actor_data, num_blocks)

        # --- Critic/value datasets: per-policy action_mask mixed sampling ---
        learner_data = self._prepare_sampled_batch(self.replay_buffer.action_mask_sample(int(self.args["learner_critic_batch_size"]), 0,on_policy_ratio=self.replay_ratio))
        learner_mini_batches = reshape_into_blocks(learner_data, num_blocks)

        oracle_mini_batches = {}
        for oracle_idx, name in enumerate(self.oracles_names):
            odata = self._prepare_sampled_batch(self.replay_buffer.action_mask_sample(int(self.args["learner_critic_batch_size"]), oracle_idx + 1,on_policy_ratio=self.replay_ratio))
            oracle_mini_batches[name] = reshape_into_blocks(odata, num_blocks)

        info = {}
        for i in range(num_blocks):
            self.update_step += 1
            # Each mini_batches is a dict of tensors, shaped (num_blocks, ...)
            learner_sub_batch = {k: v[i] for k, v in learner_mini_batches.items()}
            oracle_sub_batches = None
            oracle_sub_batches = {name: {k: v[i] for k, v in mb.items()} for name, mb in oracle_mini_batches.items()}

            critic_info = self.update_critic(learner_sub_batch, oracle_sub_batches)
            info.update(critic_info)
            self.update_target_learner_critic()
            self.update_target_oracles_critic()
                
            if self.update_step % self.policy_delay == 0:
                actor_sub_batch = {k: v[i] for k, v in actor_mini_batches.items()}
                actor_info = self.update_learner_actor(actor_sub_batch)
                info.update(actor_info)
                self.update_target_learner_actor()
        return info
    
    # Update all critics 
    def update_critic(self, learner_data: dict, oracle_data_by_name: Optional[Dict[str, dict]] = None):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        q_loss, v_loss, info = self._compute_critic_loss(learner_data, oracle_data_by_name)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        
        for name in self.oracles_names:
            self.oracles_critic_optimizers[name].zero_grad(set_to_none=True)
            self.oracles_value_optimizers[name].zero_grad(set_to_none=True)
        total_loss = q_loss + v_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.learner_critic.parameters(), self.args["max_grad_norm"])
        nn.utils.clip_grad_norm_(self.learner_value.parameters(), self.args["max_grad_norm"])
        self.critic_optimizer.step()
        self.value_optimizer.step()
        # L2 normalize after update (SIMBA-style)
        l2normalize_network(self.learner_critic)
        l2normalize_network(self.learner_value)
        
        for name in self.oracles_names:
            nn.utils.clip_grad_norm_(self.oracles_critics[name].parameters(), self.args["max_grad_norm"])
            nn.utils.clip_grad_norm_(self.oracles_values[name].parameters(), self.args["max_grad_norm"])
            self.oracles_critic_optimizers[name].step()
            self.oracles_value_optimizers[name].step()
            l2normalize_network(self.oracles_critics[name])
            l2normalize_network(self.oracles_values[name]) 
        return info

    def _compute_critic_loss(self, learner_data: dict, oracle_data_by_name: Optional[Dict[str, dict]] = None):
        ## learner part 
        info = {}
        total_q_loss = 0.0
        total_v_loss = 0.0
        q_loss, v_loss, q_mean, v_mean, q_uncertainty = self._compiled_compute_learner_critic_loss(learner_data)
        total_q_loss += q_loss
        total_v_loss += v_loss
        info.update({
            "learner/losses/qf_loss": q_loss.item(),
            "learner/losses/vf_loss": v_loss.item(),
            "learner/values/qf_mean": q_mean.item(),
            "learner/values/vf_mean": v_mean.item(),
            "learner/values/qf_uncertainty": q_uncertainty.item(),
        })
        ## oracles part
        for name in self.oracles_names:
            q_loss, v_loss, q_mean, v_mean, q_uncertainty = self._compiled_compute_oracles_critic_loss(oracle_data_by_name[name], name)
            total_q_loss += q_loss
            total_v_loss += v_loss
            info.update(
                    {
                        f"oracles/{name}/qf_loss": q_loss.item(),
                        f"oracles/{name}/vf_loss": v_loss.item(),
                        f"oracles/{name}/qf_mean": q_mean.item(),
                        f"oracles/{name}/vf_mean": v_mean.item(),
                        f"oracles/{name}/qf_uncertainty": q_uncertainty.item(),
                    }
                )
        return total_q_loss, total_v_loss, info
    def _compute_oracles_critic_loss(self, data: dict,oracle_name:str):
        ## 1-step TD Loss for critic
        with torch.no_grad():
            ## Important to not use get_eval_action here 
            std = float(self.oracle_std_multiplier_by_name[oracle_name])
            # Oracle policies may own their own observation normalizer; prefer raw tensors when available.
            next_obs_for_oracle = data.get("next_observations_raw", data["next_observations"])
            # oracle_action = self.oracles_actors[oracle_name].get_action(next_obs_for_oracle, std)
            oracle_action = self.oracles_actors[oracle_name].get_action(next_obs_for_oracle, self.target_policy_noise)
            next_qs, next_q_infos = self.oracles_critics[oracle_name].get_target_value_with_info(
                data["next_observations"], oracle_action
            )  # shape = (N_ensemble,B)
            # Use minimum Q for target (CDQ - Clipped Double Q-learning)
            # next_qs shape: [num_qs, B, 1], next_q_infos is list of dicts with 'log_prob'
            min_idx = next_qs.squeeze(-1).argmin(dim=0)  # (B,)

            # Gather the log_probs from the min Q network
            # Stack log_probs: [num_qs, B, num_bins]
            stacked_log_probs = torch.stack([info["log_prob"] for info in next_q_infos], dim=0)
            # Select log_probs for each sample from the min Q network
            next_q_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)]

        pred_qs, pred_q_infos = self.oracles_critics[oracle_name].get_value_with_info(
            data["observations"], data["actions"]
        )  # shape = (N_ensemble,B)
        # Compute categorical TD loss for each critic
        gamma_n = float(self.gamma) ** int(self.args.get("nstep", 1))

        q_loss = 0
        td_loss_fn = categorical_td_loss
        for _i, pred_info in enumerate(pred_q_infos):
            loss = td_loss_fn(
                pred_log_probs=pred_info["log_prob"],
                target_log_probs=next_q_log_probs,
                reward=data["rewards"],
                done=data["dones"],
                gamma=gamma_n,
                num_bins=self.num_bins,
                min_v=self.min_v,
                max_v=self.max_v,
            )
            q_loss = q_loss + loss.mean()

        q_mean = pred_qs.mean()
        q_uncertainty = pred_qs.std(dim=0).mean()
        
        with torch.no_grad():
            std = float(self.oracle_std_multiplier_by_name[oracle_name])
            obs_for_oracle = data.get("observations_raw", data["observations"])
            actions = self.oracles_actors[oracle_name].get_action(obs_for_oracle, std)
            current_qs, current_q_infos = self.oracles_critics[oracle_name].get_target_value_with_info(
                data["observations"], actions
            )  # shape = (num_qs, B, 1)
            min_idx = current_qs.squeeze(-1).argmin(dim=0)  # (B,) 
            stacked_log_probs = torch.stack([info["log_prob"] for info in current_q_infos], dim=0)
            current_q_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)] 

        pred_v, pred_v_infos = self.oracles_values[oracle_name].get_value_with_info(data["observations"])  
        v_loss = 0
        for _i, pred_info in enumerate(pred_v_infos):
            loss = categorical_td_loss(
                pred_log_probs=pred_info["log_prob"],
                target_log_probs=current_q_log_probs,
                reward=torch.zeros_like(data["rewards"]),
                done=torch.zeros_like(data["dones"]),
                gamma=1.0,
                num_bins=self.num_bins,
                min_v=self.min_v,
                max_v=self.max_v,
            )
            v_loss = v_loss + loss.mean()
        oracle_critic_values = pred_v.mean()
        return q_loss, v_loss, q_mean, oracle_critic_values.mean(), q_uncertainty.mean()
    
    def _compute_learner_critic_loss(self, data: dict):
        """Compute categorical TD loss for critic (SIMBA-style)."""
        # Get next actions from target policy with noise for smoothing
        with torch.no_grad():
            next_actions = self.learner_actor.get_target_action(data["next_observations"], self.target_policy_noise)

            # Get target Q values with info (stacked: [num_qs, B, 1])
            next_qs, next_q_infos = self.learner_critic.get_target_value_with_info(data["next_observations"], next_actions)

            # Use minimum Q for target (CDQ - Clipped Double Q-learning)
            # next_qs shape: [num_qs, B, 1], next_q_infos is list of dicts with 'log_prob'
            min_idx = next_qs.squeeze(-1).argmin(dim=0)  # (B,)

            # Gather the log_probs from the min Q network
            # Stack log_probs: [num_qs, B, num_bins]
            stacked_log_probs = torch.stack([info["log_prob"] for info in next_q_infos], dim=0)
            # Select log_probs for each sample from the min Q network
            next_q_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)]

        # Get current Q values with info
        pred_qs, pred_q_infos = self.learner_critic.get_value_with_info(data["observations"], data["actions"])

        # Compute categorical TD loss for each critic
        gamma_n = float(self.gamma) ** int(self.args.get("nstep", 1))

        q_loss = 0
        td_loss_fn = categorical_td_loss
        for _i, pred_info in enumerate(pred_q_infos):
            loss = td_loss_fn(
                pred_log_probs=pred_info["log_prob"],
                target_log_probs=next_q_log_probs,
                reward=data["rewards"],
                done=data["dones"],
                gamma=gamma_n,
                num_bins=self.num_bins,
                min_v=self.min_v,
                max_v=self.max_v,
            )
            q_loss = q_loss + loss.mean()

        q_mean = pred_qs.mean()
        q_uncertainty = pred_qs.std(dim=0).mean()
        
        
        
        with torch.no_grad():
            actions = self.learner_actor.get_action(data["observations"], self.target_policy_noise)
            current_qs, current_q_infos = self.learner_critic.get_target_value_with_info(
                data["observations"], actions
            )  # shape = (num_qs, B, 1)
            min_idx = current_qs.squeeze(-1).argmin(dim=0)  # (B,) 
            stacked_log_probs = torch.stack([info["log_prob"] for info in current_q_infos], dim=0)
            current_q_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)] 

        pred_v, pred_v_infos = self.learner_value.get_value_with_info(data["observations"])  
        v_loss = 0
        for _i, pred_info in enumerate(pred_v_infos):
            loss = categorical_td_loss(
                pred_log_probs=pred_info["log_prob"],
                target_log_probs=current_q_log_probs,
                reward=torch.zeros_like(data["rewards"]),
                done=torch.zeros_like(data["dones"]),
                gamma=1.0,
                num_bins=self.num_bins,
                min_v=self.min_v,
                max_v=self.max_v,
            )
            v_loss = v_loss + loss.mean()
        v_values = pred_v.mean()
        return q_loss, v_loss, q_mean, v_values.mean(), q_uncertainty
        
    # Update learner actor 
    
    def _compute_learner_actor_loss(self, data: dict):
        """
        - In-support Generalized Advantage IQL loss term  
        1.Given insupport states and actions, compute the maxQ - maxV as unormalized actor weight
        2. Sample from the behavior policy to compute the denominator of the actor weight
        3. Perform AWBC to update the actor 
    
        """
        # In-support advantage computed incrementally
        actions = data["actions"]  # (B, A)

        with torch.no_grad():
            qs = self.learner_critic.get_value(data["observations"], actions)  # (num_qs, B, 1)
            q_max = qs.mean(0).squeeze(-1)  # (B,)
            vs = self.learner_value.get_value(data["observations"])  # (num_qs, B, 1)
            v_max = vs.mean(0).squeeze(-1)  # (B,)
            for critic in self._oracle_critic_list:
                qk = critic.get_value(data["observations"], actions).squeeze(-1).mean(dim=0)
                q_max = torch.maximum(q_max, qk)
            for vnet in self._oracle_value_list:
                vk = vnet.get_value(data["observations"]).squeeze(-1).mean(dim=0)
                v_max = torch.maximum(v_max, vk)
            advantage = q_max - v_max
            pos_mask = (advantage > 0)  # (B,)
            pos_mask_f = pos_mask.to(dtype=advantage.dtype)
            positive_ratio = pos_mask_f.mean()
            weights = torch.exp((self.beta * advantage).clamp(-50.0, 10.0)) * pos_mask_f  # (B,)
        eps = 1e-8
        pos_count = pos_mask_f.sum()
        # Normalize weights over the positive-advantage subset (avoid NaNs when no positives)
        mean_weight = weights.sum() / (pos_count + eps)
        normalized_weights = weights / (mean_weight + eps)  # (B,)
        predicted_action = self.learner_actor.get_eval_action(data["observations"])  # (B, A)
        raw_loss = (predicted_action - data["actions"]).pow(2).sum(dim=-1)
        awbc_loss = (normalized_weights.detach() * raw_loss).sum() / (pos_count + eps)

        # Neutral TD3 term
        a_pi = self.learner_actor.get_eval_action(data["observations"])  # (B, A)
        q_pi = self.learner_critic.get_value(data["observations"], a_pi)  # (num_qs, B, 1)
        q_pi = q_pi.mean(0).squeeze(-1)  # (B,)
        td3_obj = (-1.0 * q_pi).mean()

        return awbc_loss, td3_obj, q_pi.mean(), weights.mean(), positive_ratio
    def get_actor_loss_weight(self):
        
        if self.use_adaptive_actor_loss_weight:
            arr = np.array(self.action_selection_buffer, dtype=int)
            learner_ratio = float((arr == 0).mean())
            bc_weigth_multiplier = (1 - learner_ratio) ## if learner ratio is close to 1.0, we expect the BC close to 0.0 
            bc_weight = bc_weigth_multiplier * self.l1_loss_weight 
            rl_weight = self.rl_loss_weight
            
        else:
            bc_weight = self.l1_loss_weight
            rl_weight = self.rl_loss_weight
            
        return {
                'rl_weight':rl_weight,
                'bc_weight':bc_weight
            }
    
    
    def update_learner_actor(self,data:dict):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        awbc_loss,rl_loss,q_pi_mean,actor_weight,weight_positive_ratio = self._compiled_compute_learner_actor_loss(data)
        
        # Backprop once on a combined loss to avoid double-backward through shared graphs
        self.actor_optimizer.zero_grad(set_to_none=True)
        weights = self.get_actor_loss_weight()
        total_actor_loss = rl_loss * weights['rl_weight'] + awbc_loss * weights['bc_weight']
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.learner_actor.parameters(), self.args["policy_max_grad_norm"])
        self.actor_optimizer.step()
        
        # L2 normalize after update (SIMBA-style)
        l2normalize_network(self.learner_actor)
        
        actor_info = {
            "learner/losses/qf_mean": q_pi_mean.item(),
            "learner/losses/rl_loss": rl_loss.item(),
            "learner/losses/awbc_loss": awbc_loss.item(),
            "learner/losses/rl_weight": weights['rl_weight'],
            "learner/losses/bc_weight": weights['bc_weight'],
            "learner/losses/maxadv_weight": actor_weight.item(),
            "learner/losses/positive_ratio": weight_positive_ratio.item(),
        }
        return actor_info

    
    
    
    
    def update_target_oracles_critic(self):
        for oracle_name in self.oracles_names:
            with torch.no_grad():
                self.oracles_critics[oracle_name].update_target(self.args["tau"])
                self.oracles_values[oracle_name].update_target(self.args["tau"])
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
                "oracles_names": list(self.oracles_names),
            }
        )

        for i, name in enumerate(self.oracles_names):
            checkpoint[f"oracles_actor_state_dict_{i}"] = self.oracles_actors[name].state_dict()
            checkpoint[f"oracles_critic_state_dict_{i}"] = self.oracles_critics[name].state_dict()
            checkpoint[f"oracles_value_state_dict_{i}"] = self.oracles_values[name].state_dict()
            checkpoint[f"oracles_critic_optimizer_state_dict_{i}"] = self.oracles_critic_optimizers[name].state_dict()
            checkpoint[f"oracles_value_optimizer_state_dict_{i}"] = self.oracles_value_optimizers[name].state_dict()

        # RNG + sampling buffer
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

        saved_names = checkpoint.get("oracles_names", list(self.oracles_names))
        if list(saved_names) != list(self.oracles_names):
            print("[MaxAdv] Warning: oracles_names mismatch between checkpoint and current model. Loading in current order.")

        for i, name in enumerate(self.oracles_names):
            k_actor = f"oracles_actor_state_dict_{i}"
            k_critic = f"oracles_critic_state_dict_{i}"
            k_value = f"oracles_value_state_dict_{i}"
            if k_actor in checkpoint:
                self.oracles_actors[name].load_state_dict(checkpoint[k_actor])
            if k_critic in checkpoint:
                self.oracles_critics[name].load_state_dict(checkpoint[k_critic])
            if k_value in checkpoint:
                self.oracles_values[name].load_state_dict(checkpoint[k_value])

            k_copt = f"oracles_critic_optimizer_state_dict_{i}"
            k_vopt = f"oracles_value_optimizer_state_dict_{i}"
            if k_copt in checkpoint:
                self.oracles_critic_optimizers[name].load_state_dict(checkpoint[k_copt])
            if k_vopt in checkpoint:
                self.oracles_value_optimizers[name].load_state_dict(checkpoint[k_vopt])

        # Restore RNG states
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
            print(f"[MaxAdv] Warning: failed to restore RNG state: {e}")

        # Restore sampling buffer
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
