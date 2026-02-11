import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import TanhTransform
import numpy as np
from model.simba_base import CategoricalCritic, CategoricalValue, DeterministicPolicy


class EnsembleCritic(nn.Module):
    
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
        num_qs: int = 2
    ):
        super().__init__()
        self.critics = nn.ModuleList([
            CategoricalCritic(
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
                max_v=max_v
            ) for _ in range(num_qs)
        ])

    def forward(self, observations, actions):
        # Run all critics
        qs_list = []
        infos_list = []
        
        for critic in self.critics:
            q, info = critic(observations, actions)
            qs_list.append(q)
            infos_list.append(info)
            
        # Stack to get shape [num_qs, Batch, 1]
        qs = torch.stack(qs_list, dim=0)
        
        # Info merging (optional, depends on how you use it)
        infos = infos_list # Simplified return
        
        return qs, infos


class EnsembleValue(nn.Module):
    
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
        num_qs: int = 2
    ):
        super().__init__()
        self.values = nn.ModuleList([
            CategoricalValue(
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
                max_v=max_v
            ) for _ in range(num_qs)
        ])

    def forward(self, observations):
        # Run all critics
        vs_list = []
        infos_list = []
        
        for value in self.values:
            v, info = value(observations)
            vs_list.append(v)
            infos_list.append(info)
            
        # Stack to get shape [num_qs, Batch, 1]
        vs = torch.stack(vs_list, dim=0)
        
        # Info merging (optional, depends on how you use it)
        infos = infos_list # Simplified return
        
        return vs, infos


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
    


class SimbaCritics(nn.Module):
    def __init__(self, 
                 obs_shape, 
                 action_dim,
                 model_args):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_dim = obs_shape[0]
        self.action_dim = action_dim
        self.critic = EnsembleCritic(
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
        qs, _ = self.critic(observations, actions)
        return qs 

    def get_target_value(self, observations: torch.Tensor, actions: torch.Tensor):
        qs, _ = self.critic_target(observations, actions)
        return qs 

    def get_value_with_info(self, observations: torch.Tensor, actions: torch.Tensor):
        """Get Q values and info dict containing log_probs for categorical TD loss."""
        qs, infos = self.critic(observations, actions)
        return qs, infos

    def get_target_value_with_info(self, observations: torch.Tensor, actions: torch.Tensor):
        """Get target Q values and info dict containing log_probs for categorical TD loss."""
        qs, infos = self.critic_target(observations, actions)
        return qs, infos

    def update_target(self, tau=0.005):
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class SimbaValues(nn.Module):
    def __init__(self, 
                 obs_shape, 
                 action_dim,
                 model_args):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_dim = obs_shape[0]
        self.action_dim = action_dim
        self.value = EnsembleValue(
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
        vs, _ = self.value(observations)
        return vs 

    def get_target_value(self, observations: torch.Tensor):
        vs, _ = self.value_target(observations)
        return vs 
    
    def get_value_with_info(self, observations: torch.Tensor):
        vs, infos = self.value(observations)
        return vs, infos

    def get_target_value_with_info(self, observations: torch.Tensor):
        vs, infos = self.value_target(observations)
        return vs, infos

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