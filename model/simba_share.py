import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.simba_base import  DeterministicPolicy, HyperEmbedder, HyperLERPBlock, HyperCategoricalValue


class SharedCategoricalCritic(nn.Module):
    def __init__(
        self,
        input_dim: int, # Obs dim + Action dim
        num_blocks: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        num_bins: int,
        min_v: float,
        max_v: float,
        n_heads: int = 1,
    ):
        super().__init__()
        
        self.embedders = nn.ModuleList([
            HyperEmbedder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                c_shift=c_shift,
            ) for _ in range(n_heads)
        ])
            
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
            )
        self.encoder = nn.Sequential(*blocks)

        self.predictors = nn.ModuleList([
            HyperCategoricalValue(
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        ) for _ in range(n_heads)])

    def forward_single_head(self, observations, actions, head_idx):
        x = torch.cat([observations, actions], dim=-1)
        y = self.embedders[head_idx](x)
        z = self.encoder(y)
        q, info = self.predictors[head_idx](z)
        return q, info

    def forward(self, observations, actions):
        qs_list = []
        infos_list = []
        x = torch.cat([observations, actions], dim=-1)
        for head_idx, (embedder, predictor) in enumerate(zip(self.embedders, self.predictors)):
            y = embedder(x)
            z = self.encoder(y)
            q, info = predictor(z)
            qs_list.append(q)
            infos_list.append(info)
        return torch.stack(qs_list, dim=0), infos_list

class SharedCategoricalValue(nn.Module):
    def __init__(
        self,
        input_dim: int, # Obs dim + Action dim
        num_blocks: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        num_bins: int,
        min_v: float,
        max_v: float,
        n_heads: int = 1,
    ):
        super().__init__()
        
        self.embedders = nn.ModuleList([
            HyperEmbedder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                c_shift=c_shift,
            ) for _ in range(n_heads)
        ])
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
            )
        self.encoder = nn.Sequential(*blocks)

        self.predictors = nn.ModuleList([
            HyperCategoricalValue(
            hidden_dim=hidden_dim,
            num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
                scaler_init=1.0,
                scaler_scale=1.0,
            ) for _ in range(n_heads)
        ])

    def forward_single_head(self, observations, head_idx):
        x = observations
        y = self.embedders[head_idx](x)
        z = self.encoder(y)
        value, info = self.predictors[head_idx](z)
        return value, info
    
    def forward(self, observations):
        values_list = []
        infos_list = []
        for head_idx, (embedder, predictor) in enumerate(zip(self.embedders, self.predictors)):
            y = embedder(observations)
            z = self.encoder(y)
            value, info = predictor(z)
            values_list.append(value)
            infos_list.append(info)
        return torch.stack(values_list, dim=0), infos_list




class SharedEnsembleCritic(nn.Module):
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        num_blocks: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
        n_heads: int = 1,
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.num_qs = int(num_qs)
        self.critics = nn.ModuleList([
            SharedCategoricalCritic(
                input_dim=observation_dim + action_dim,
                num_blocks=num_blocks,
                hidden_dim=hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                c_shift=c_shift,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
                n_heads=n_heads,
            ) for _ in range(num_qs)
        ])

    def forward_single_head(self, observations, actions, head_idx):
        """
        Return ensemble Q estimates for a *single* head.

        Returns:
          - qs: (num_qs, B, 1)
          - infos: list[dict] of length num_qs (each dict contains e.g. "log_prob")
        """
        head_idx = int(head_idx)
        qs_list = []
        infos_list = []
        for critic in self.critics:
            q, info = critic.forward_single_head(observations, actions, head_idx)
            qs_list.append(q)
            infos_list.append(info)
        return torch.stack(qs_list, dim=0), infos_list

    def forward(self, observations, actions, head_idx: int = 0):
        """
        Convenience wrapper mirroring `model/simba.py`'s EnsembleCritic API.
        Defaults to head_idx=0.
        """
        return self.forward_single_head(observations, actions, head_idx=head_idx)

    def forward_all_heads(self, observations, actions):
        """
        Evaluate all heads.

        Returns:
          - qs: (n_heads, num_qs, B, 1)
          - infos: list[list[dict]] with shape [n_heads][num_qs]
        """
        qs_by_head = []
        infos_by_head = []
        for h in range(self.n_heads):
            q, info = self.forward_single_head(observations, actions, head_idx=h)
            qs_by_head.append(q)
            infos_by_head.append(info)
        return torch.stack(qs_by_head, dim=0), infos_by_head


class SharedEnsembleValue(nn.Module):
    
    def __init__(
        self,
        observation_dim: int,
        num_blocks: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
        n_heads: int = 1,
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.num_qs = int(num_qs)
        self.values = nn.ModuleList([
            SharedCategoricalValue(
                input_dim=observation_dim,
                num_blocks=num_blocks,
                hidden_dim=hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                c_shift=c_shift,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
                n_heads=n_heads,
            ) for _ in range(num_qs)
        ])

    def forward_single_head(self, observations, head_idx):
        """
        Return ensemble V estimates for a *single* head.

        Returns:
          - vs: (num_qs, B, 1)
          - infos: list[dict] of length num_qs (each dict contains e.g. "log_prob")
        """
        values_list = []
        infos_list = []
        for value in self.values:
            v, info = value.forward_single_head(observations, head_idx)
            values_list.append(v)
            infos_list.append(info)
        return torch.stack(values_list, dim=0), infos_list

    def forward(self, observations, head_idx: int = 0):
        """
        Convenience wrapper mirroring `model/simba.py`'s EnsembleValue API.
        Defaults to head_idx=0.
        """
        return self.forward_single_head(observations, head_idx=head_idx)

    def forward_all_heads(self, observations):
        """
        Evaluate all heads.

        Returns:
          - vs: (n_heads, num_qs, B, 1)
          - infos: list[list[dict]] with shape [n_heads][num_qs]
        """
        vs_by_head = []
        infos_by_head = []
        for h in range(self.n_heads):
            v, info = self.forward_single_head(observations, head_idx=h)
            vs_by_head.append(v)
            infos_by_head.append(info)
        return torch.stack(vs_by_head, dim=0), infos_by_head


# -----------------------------------------
# Agent Interface
# -----------------------------------------

LOG_STD_MAX = 2
LOG_STD_MIN = -5
from torch.distributions.utils import _standard_normal
class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-5):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        # Clamp to [low+eps, high-eps] for numerical stability
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        # Use the "straight-through" trick for gradients
        return x - x.detach() + clamped_x.detach()

    def rsample(self, sample_shape=torch.Size(), clip=None):
        # Reparameterized sample
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = torch.tanh(self.loc) + eps
        return self._clamp(x)

    def sample(self, sample_shape=torch.Size(), clip=None):
        # Non-reparameterized sample
        with torch.no_grad():
            return self.rsample(sample_shape, clip=clip)

    @property
    def mean_truncated(self):
        # Approximate mean of the truncated normal
        # See: https://en.wikipedia.org/wiki/Truncated_normal_distribution#Truncation_to_a_finite_interval
        mean = torch.tanh(self.loc) 
        return mean
    


class SharedSimbaCritics(nn.Module):
    def __init__(self, 
                 obs_shape, 
                 action_dim,
                 model_args):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_dim = obs_shape[0]
        self.action_dim = action_dim
        self.n_heads = int(getattr(model_args, "n_heads", 1))
        self.critic = SharedEnsembleCritic(
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_blocks=model_args.num_blocks,
            hidden_dim=model_args.hidden_dim,
            scaler_init=model_args.scaler_init,
            scaler_scale=model_args.scaler_scale,
            alpha_init=model_args.alpha_init,
            alpha_scale=model_args.alpha_scale,
            c_shift=model_args.c_shift,
            num_bins=model_args.num_bins,
            min_v=model_args.min_v,
            max_v=model_args.max_v,
            num_qs=model_args.num_qs,
            n_heads=model_args.n_heads,
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def set_optimizer(self, critic_lr, weight_decay=1e-2):
        critic_params = list(self.critic.parameters())
        extra_args = {"foreach": True}
        return torch.optim.AdamW(critic_params, lr=critic_lr, weight_decay=weight_decay, **extra_args)
           

    def get_value(self, observations: torch.Tensor, actions: torch.Tensor):
        """Back-compat: returns head_idx=0."""
        qs, _ = self.critic(observations, actions, head_idx=0)
        return qs 

    def get_target_value(self, observations: torch.Tensor, actions: torch.Tensor):
        """Back-compat: returns head_idx=0."""
        qs, _ = self.critic_target(observations, actions, head_idx=0)
        return qs 

    def get_value_with_info(self, observations: torch.Tensor, actions: torch.Tensor):
        """Back-compat: returns head_idx=0."""
        qs, infos = self.critic(observations, actions, head_idx=0)
        return qs, infos

    def get_target_value_with_info(self, observations: torch.Tensor, actions: torch.Tensor):
        """Back-compat: returns head_idx=0."""
        qs, infos = self.critic_target(observations, actions, head_idx=0)
        return qs, infos

    # ---- Head-specific APIs (preferred) ----
    def get_value_single_head(self, observations: torch.Tensor, actions: torch.Tensor, head_idx: int):
        qs, _ = self.critic(observations, actions, head_idx=int(head_idx))
        return qs

    def get_target_value_single_head(self, observations: torch.Tensor, actions: torch.Tensor, head_idx: int):
        qs, _ = self.critic_target(observations, actions, head_idx=int(head_idx))
        return qs

    def get_value_with_info_single_head(self, observations: torch.Tensor, actions: torch.Tensor, head_idx: int):
        qs, infos = self.critic(observations, actions, head_idx=int(head_idx))
        return qs, infos

    def get_target_value_with_info_single_head(self, observations: torch.Tensor, actions: torch.Tensor, head_idx: int):
        qs, infos = self.critic_target(observations, actions, head_idx=int(head_idx))
        return qs, infos

    def update_target(self, tau=0.005):
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class SharedSimbaValues(nn.Module):
    def __init__(self, 
                 obs_shape, 
                 action_dim,
                 model_args):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_dim = obs_shape[0]
        self.action_dim = action_dim
        self.n_heads = int(getattr(model_args, "n_heads", 1))
        self.value = SharedEnsembleValue(
            observation_dim=self.obs_dim,
            num_blocks=model_args.num_blocks,
            hidden_dim=model_args.hidden_dim,
            scaler_init=model_args.scaler_init,
            scaler_scale=model_args.scaler_scale,
            alpha_init=model_args.alpha_init,
            alpha_scale=model_args.alpha_scale,
            c_shift=model_args.c_shift,
            num_bins=model_args.num_bins,
            min_v=model_args.min_v,
            max_v=model_args.max_v,
            num_qs=model_args.num_qs,
            n_heads=model_args.n_heads,
        )
        self.value_target = copy.deepcopy(self.value)
        self.value_target.eval()
        for param in self.value_target.parameters():
            param.requires_grad = False

    def set_optimizer(self, critic_lr, weight_decay=1e-2):
        value_params = list(self.value.parameters())
        extra_args = {"foreach": True}
        return torch.optim.AdamW(value_params, lr=critic_lr, weight_decay=weight_decay, **extra_args)

    def get_value(self, observations: torch.Tensor):
        """Back-compat: returns head_idx=0."""
        vs, _ = self.value(observations, head_idx=0)
        return vs 

    def get_target_value(self, observations: torch.Tensor):
        """Back-compat: returns head_idx=0."""
        vs, _ = self.value_target(observations, head_idx=0)
        return vs 

    # ---- Head-specific APIs (preferred) ----
    def get_value_single_head(self, observations: torch.Tensor, head_idx: int):
        vs, _ = self.value(observations, head_idx=int(head_idx))
        return vs

    def get_target_value_single_head(self, observations: torch.Tensor, head_idx: int):
        vs, _ = self.value_target(observations, head_idx=int(head_idx))
        return vs

    def update_target(self, tau=0.005):
        with torch.no_grad():
            for target_param, param in zip(self.value_target.parameters(), self.value.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

class DeterministicSimbaPolicy(nn.Module):
    def __init__(self, 
                 obs_shape, 
                 action_dim,
                 model_args):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_dim = obs_shape[0]
        self.action_dim = action_dim
        self.noise_clip = model_args.noise_clip
        self.policy = DeterministicPolicy(
            input_dim=self.obs_dim,
            num_blocks=model_args.num_blocks,
            hidden_dim=model_args.hidden_dim,
            action_dim=self.action_dim,
            scaler_init=model_args.scaler_init,
            scaler_scale=model_args.scaler_scale,
            alpha_init=model_args.alpha_init,
            alpha_scale=model_args.alpha_scale,
            c_shift=model_args.c_shift,
        )
        self.policy_target = copy.deepcopy(self.policy)
        self.policy_target.eval()
        for param in self.policy_target.parameters():
            param.requires_grad = False

        self._register_action_bounds(model_args)

    def _register_action_bounds(self, model_args):
        if getattr(model_args, "action_high", None) is not None and getattr(model_args, "action_low", None) is not None:
            action_high = torch.as_tensor(model_args.action_high, dtype=torch.float32)
            action_low = torch.as_tensor(model_args.action_low, dtype=torch.float32)
        else:
            action_high = torch.ones(self.action_dim, dtype=torch.float32)
            action_low = -torch.ones(self.action_dim, dtype=torch.float32)

        self.register_buffer("action_high", action_high)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def set_optimizer(self, actor_lr, weight_decay=1e-2):
        actor_params = list(self.policy.parameters())
        extra_args = {"foreach": True}
        return torch.optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay, **extra_args)



    def get_target_action(self, observations: torch.Tensor, std: float):
        means = self.policy_target(observations)
        truncated_normal = TruncatedNormal(means, std)
        action = truncated_normal.rsample(clip=self.noise_clip)
        return action 

    def get_action(self, observations: torch.Tensor, std: float):
        means = self.policy(observations)
        truncated_normal = TruncatedNormal(means, std)
        action = truncated_normal.rsample(clip=self.noise_clip)
        return action

    def sample_action(self, observations: torch.Tensor, std: float, n_actions: int):
        means = self.policy(observations)
        truncated_normal = TruncatedNormal(means, std)
        action = truncated_normal.rsample(sample_shape=(n_actions,), clip=self.noise_clip)
        return action

    def get_eval_action(self, observations: torch.Tensor):
        means = self.policy(observations)
        truncated_normal = TruncatedNormal(means, 1.0)
        return truncated_normal.mean_truncated

    def get_target_eval_action(self, observations: torch.Tensor):
        means = self.policy_target(observations)
        truncated_normal = TruncatedNormal(means, 1.0)
        return truncated_normal.mean_truncated

    def update_target(self, tau=0.005):
        with torch.no_grad():
            for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)