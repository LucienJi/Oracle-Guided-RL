import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_spectral_norm(layer: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return layer
    # Prefer parametrizations API when available (PyTorch >= 1.12).
    try:
        from torch.nn.utils.parametrizations import spectral_norm as _sn
        return _sn(layer)
    except Exception:
        from torch.nn.utils import spectral_norm as _sn_legacy
        return _sn_legacy(layer)


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """
    Flexible MLP with optional residual connections and normalization.

    Normalization options:
      - norm="none": no activation normalization
      - norm="layernorm": LayerNorm after each hidden linear
      - norm="l2": L2-normalize hidden activations (F.normalize)
    Spectral normalization:
      - spectral_norm=True wraps Linear layers with spectral norm.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        activation: str = "relu",
        norm: str = "none",
        spectral_norm: bool = False,
        residual: bool = False,
        residual_projection: bool = False,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = [int(x) for x in hidden_dims]
        self.norm = str(norm).lower()
        self.residual = bool(residual)
        self.residual_projection = bool(residual_projection)
        self.eps = float(eps)

        act = _activation(activation)
        self._act_name = activation

        dims = [self.input_dim] + list(self.hidden_dims)
        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            lin = _maybe_spectral_norm(nn.Linear(in_d, out_d), spectral_norm)
            self.fcs.append(lin)
            if self.norm == "layernorm":
                self.norms.append(nn.LayerNorm(out_d))
            else:
                self.norms.append(nn.Identity())
            if self.residual and self.residual_projection and in_d != out_d:
                self.res_projs.append(nn.Linear(in_d, out_d, bias=False))
            else:
                self.res_projs.append(nn.Identity())
            self.dropouts.append(nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity())

        self.activation = act
        self.out = _maybe_spectral_norm(nn.Linear(dims[-1], self.output_dim), spectral_norm)

        # Init: orthogonal-ish tends to be stable for RL MLPs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.out.weight, gain=1e-2)
        if self.out.bias is not None:
            nn.init.constant_(self.out.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the penultimate (pre-head) features.
        This is used for representation diagnostics (e.g., effective rank / SVD metrics).
        """
        h = x
        for fc, norm, proj, do in zip(self.fcs, self.norms, self.res_projs, self.dropouts):
            prev = h
            h = fc(h)
            h = norm(h)
            h = self.activation(h)
            if self.norm == "l2":
                h = F.normalize(h, dim=-1, eps=self.eps)
            h = do(h)
            if self.residual:
                if prev.shape == h.shape:
                    h = h + prev
                elif self.residual_projection:
                    h = h + proj(prev)
        return h

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        h = self.forward_features(x)
        y = self.out(h)
        if return_features:
            return y, h
        return y


class SquashedNormal:
    """
    Tanh-squashed Normal distribution helper with stable log-prob correction.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.normal = torch.distributions.Normal(mean, std)
        self.eps = float(eps)

    def rsample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.normal.rsample()
        a = torch.tanh(z)
        # log_prob = log N(z) - log|det(d tanh(z)/dz)|
        log_prob = self.normal.log_prob(z)
        log_prob = log_prob - torch.log(1.0 - a.pow(2) + self.eps)
        return a, log_prob.sum(dim=-1, keepdim=True)

    def mean_action(self) -> torch.Tensor:
        return torch.tanh(self.mean)


@dataclass
class ActorMLPConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    norm: str = "none"  # none|layernorm|l2
    spectral_norm: bool = False
    residual: bool = False
    residual_projection: bool = False
    dropout: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    action_low: Optional[Sequence[float]] = None
    action_high: Optional[Sequence[float]] = None


@dataclass
class CriticMLPConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    norm: str = "none"  # none|layernorm|l2
    spectral_norm: bool = False
    residual: bool = False
    residual_projection: bool = False
    dropout: float = 0.0


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_dim: int, model_args: Any):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))
        self.obs_dim = obs_dim
        self.action_dim = int(action_dim)

        # Expect dict-like (e.g., DictConfig/dict) or an object with __dict__.
        cfg = model_args if hasattr(model_args, "__getitem__") else getattr(model_args, "__dict__", model_args)
        hidden_dims = tuple(int(x) for x in cfg["hidden_dims"])
        if len(hidden_dims) == 0:
            raise ValueError("SquashedGaussianActor: 'hidden_dims' must be non-empty.")
        activation = str(cfg["activation"])
        norm = str(cfg["norm"])
        spectral = bool(cfg["spectral_norm"])
        residual = bool(cfg["residual"])
        residual_proj = bool(cfg["residual_projection"])
        dropout = float(cfg["dropout"])
        self.log_std_min = float(cfg["log_std_min"])
        self.log_std_max = float(cfg["log_std_max"])
      

        self.trunk = MLP(
            input_dim=self.obs_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],
            activation=activation,
            norm=norm,
            spectral_norm=spectral,
            residual=residual,
            residual_projection=residual_proj,
            dropout=dropout,
        )
        # Replace trunk.out with identity; use explicit heads for mean/log_std
        self.trunk.out = nn.Identity()

        self.mean = nn.Linear(hidden_dims[-1], self.action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], self.action_dim)
        nn.init.orthogonal_(self.mean.weight, gain=1e-2)
        nn.init.constant_(self.mean.bias, 0.0)
        nn.init.orthogonal_(self.log_std.weight, gain=1e-2)
        nn.init.constant_(self.log_std.bias, 0.0)

        self._register_action_bounds(cfg)

    def _register_action_bounds(self, cfg: Dict[str, Any]) -> None:
        ah = cfg.get("action_high", None)
        al = cfg.get("action_low", None)
        if ah is None or al is None:
            ah_t = torch.ones(self.action_dim, dtype=torch.float32)
            al_t = -torch.ones(self.action_dim, dtype=torch.float32)
        else:
            ah_t = torch.as_tensor(ah, dtype=torch.float32)
            al_t = torch.as_tensor(al, dtype=torch.float32)
        self.register_buffer("action_high", ah_t)
        self.register_buffer("action_low", al_t)
        self.register_buffer("action_scale", (ah_t - al_t) / 2.0)
        self.register_buffer("action_bias", (ah_t + al_t) / 2.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean(h)
        log_std = self.log_std(h)
        # squashed to [log_std_min, log_std_max]
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (1 + torch.tanh(log_std))
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(obs)
        dist = SquashedNormal(mean, std)
        a, logp = dist.rsample()
        return a, logp

    def eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _std = self.forward(obs)
        return torch.tanh(mean)


class DoubleQCritic(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_dim: int, model_args: Any):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))
        self.obs_dim = obs_dim
        self.action_dim = int(action_dim)

        cfg = model_args if hasattr(model_args, "__getitem__") else getattr(model_args, "__dict__", model_args)
        hidden_dims = tuple(int(x) for x in cfg["hidden_dims"])
        if len(hidden_dims) == 0:
            raise ValueError("DoubleQCritic: 'hidden_dims' must be non-empty.")
        activation = str(cfg["activation"])
        norm = str(cfg["norm"])
        spectral = bool(cfg["spectral_norm"])
        residual = bool(cfg["residual"])
        residual_proj = bool(cfg["residual_projection"])
        dropout = float(cfg["dropout"])

        in_dim = self.obs_dim + self.action_dim
        self.q1 = MLP(
            input_dim=in_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            norm=norm,
            spectral_norm=spectral,
            residual=residual,
            residual_projection=residual_proj,
            dropout=dropout,
        )
        self.q2 = MLP(
            input_dim=in_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            norm=norm,
            spectral_norm=spectral,
            residual=residual,
            residual_projection=residual_proj,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
        


class CriticWithTarget(nn.Module):
    """
    Convenience wrapper matching the pattern used in `model/simba.py`:
    exposes critic + critic_target and a soft update helper.
    """

    def __init__(self, critic: DoubleQCritic):
        super().__init__()
        self.critic = critic
        self.critic_target = copy.deepcopy(critic)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.critic(obs, act)

    @torch.no_grad()
    def target(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.critic_target(obs, act)

    @torch.no_grad()
    def update_target(self, tau: float = 0.005) -> None:
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


