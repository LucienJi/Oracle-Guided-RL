import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import TanhTransform

import numpy as np

# -----------------------------------------
# Utils & Basic Layers
# -----------------------------------------

EPS = 1e-8

def l2normalize(x, dim=-1, eps=EPS):
    """
    Standard L2 normalization: x / max(||x||, eps)
    """
    norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
    # Jax logic: x / jnp.maximum(l2norm, EPS)
    return x / torch.clamp(norm, min=eps)

import re

def l2normalize_network(
    network: nn.Module,
    eps: float = 1e-8,
    regex: str = None  # 可选：如果你只想归一化特定名称的层
) -> nn.Module:
    """
    In-place L2 normalization for all HyperDense layers in the network.
    Mimics the logic of:
    params = tree_map_until_match(f=l2normalize_layer, tree=params, target_re=regex)
    """
    
    # 既然是修改权重，我们不需要计算梯度
    with torch.no_grad():
        for name, module in network.named_modules():
            # 1. 检查是否是我们定义的 HyperDense 层
            if isinstance(module, HyperDense):
                
                # 2. (可选) 如果提供了 regex，检查模块名称是否匹配
                # PyTorch 的 name 是类似 "encoder.0.mlp.w1" 的路径
                if regex and not re.search(regex, name):
                    continue

                # 3. 执行归一化
                # 注意：PyTorch 的 nn.Linear 权重形状是 (Out, In)
                # Jax 的 nn.Dense 权重形状是 (In, Out)，且 Jax 代码用了 axis=0 (沿 Input 维度)
                # 为了数学等价，我们需要在 PyTorch 中沿 dim=1 (In 维度) 进行归一化
                
                weight = module.weight
                
                # 计算 L2 范数 (keepdim=True 保持形状为 [Out, 1])
                l2norm = torch.linalg.norm(weight, ord=2, dim=1, keepdim=True)
                
                # 原地除法 (In-place division)
                # 使用 clamp 防止除以 0
                weight.div_(torch.clamp(l2norm, min=eps))
                
    return network


class Scaler(nn.Module):
    """
    Learnable scaler parameter.
    """
    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.init_val = init
        self.scale_val = scale
        
        # self.scaler corresponds to the learnable param in Jax
        # Initialized to 1.0 * scale
        self.scaler = nn.Parameter(torch.full((dim,), 1.0 * scale))
        
        # Constant factor
        self.forward_scaler = init / scale

    def forward(self, x):
        return x * self.scaler * self.forward_scaler

class HyperDense(nn.Linear):
    """
    Dense layer without bias, using Orthogonal initialization.
    """
    def __init__(self, in_features: int, out_features: int):
        # use_bias=False in Jax code
        super().__init__(in_features, out_features, bias=False)
        # kernel_init=nn.initializers.orthogonal(scale=1.0)
        nn.init.orthogonal_(self.weight, gain=1.0)

class HyperMLP(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int, 
        scaler_init: float, 
        scaler_scale: float, 
        eps: float = 1e-8
    ):
        super().__init__()
        self.eps = eps
        
        self.w1 = HyperDense(in_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(hidden_dim, out_dim)

    def forward(self, x):
        x = self.w1(x)
        x = self.scaler(x)
        # ReLU + eps to prevent zero vector
        x = F.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, dim=-1)
        return x

class HyperEmbedder(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        scaler_init: float, 
        scaler_scale: float, 
        c_shift: float
    ):
        super().__init__()
        self.c_shift = c_shift
        
        # Input dimension increases by 1 because we concat c_shift
        self.w = HyperDense(input_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x):
        # Create new axis with c_shift value
        # x shape: [Batch, ..., Dim]
        shape = x.shape[:-1] + (1,)
        new_axis = torch.ones(shape, device=x.device, dtype=x.dtype) * self.c_shift
        
        x = torch.cat([x, new_axis], dim=-1)
        x = l2normalize(x, dim=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, dim=-1)
        return x

class HyperLERPBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        scaler_init: float, 
        scaler_scale: float, 
        alpha_init: float, 
        alpha_scale: float, 
        expansion: int = 4
    ):
        super().__init__()
        
        # MLP Input: hidden_dim -> Hidden: hidden*expansion -> Out: hidden_dim
        self.mlp = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * expansion,
            out_dim=hidden_dim,
            scaler_init=scaler_init / math.sqrt(expansion),
            scaler_scale=scaler_scale / math.sqrt(expansion),
        )
        
        self.alpha_scaler = Scaler(
            hidden_dim,
            init=alpha_init,
            scale=alpha_scale
        )

    def forward(self, x):
        residual = x
        
        # Path 1: MLP
        out = self.mlp(x)
        
        # Path 2: LERP logic
        # Jax: x = residual + self.alpha_scaler(x - residual)
        # effectively: residual + alpha * (mlp_out - residual)
        out = residual + self.alpha_scaler(out - residual)
        
        out = l2normalize(out, dim=-1)
        return out

# -----------------------------------------
# Policy (Actor) Head
# -----------------------------------------

class HyperNormalTanhPolicy(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        action_dim: int, 
        scaler_init: float, 
        scaler_scale: float,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Mean Head
        self.mean_w1 = HyperDense(hidden_dim, hidden_dim)
        self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.mean_w2 = HyperDense(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))

        # Std Head
        self.std_w1 = HyperDense(hidden_dim, hidden_dim)
        self.std_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.std_w2 = HyperDense(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x, temperature: float = 1.0):
        # Mean Calculation
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias

        # Log Std Calculation
        log_std = self.std_w1(x)
        log_std = self.std_scaler(log_std)
        log_std = self.std_w2(log_std) + self.std_bias

        # Normalize log-stds (Tanh squashing for stability)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + torch.tanh(log_std)
        )

        # Create Distribution: Tanh(Normal(mu, sigma))
        std = torch.exp(log_std) * temperature
        
        # Independent ensures that the distribution is treated as a multivariate normal
        # with diagonal covariance, rather than a batch of univariate normals.
        base_dist = Independent(Normal(mean, std), 1)
        dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

        info = {
            "mean": mean,
            "std": std,
            "log_std": log_std,
        }

        return dist, info

class HyperDeterministicPolicy(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        action_dim: int, 
        scaler_init: float, 
        scaler_scale: float,
    ):
        super().__init__()

        # Mean Head
        self.mean_w1 = HyperDense(hidden_dim, hidden_dim)
        self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.mean_w2 = HyperDense(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))


    def forward(self, x):
        # Mean Calculation
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias

        return mean

# -----------------------------------------
# Value (Critic) Head
# -----------------------------------------

class HyperCategoricalValue(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        num_bins: int, 
        min_v: float, 
        max_v: float, 
        scaler_init: float, 
        scaler_scale: float
    ):
        super().__init__()
        self.num_bins = num_bins
        
        self.w1 = HyperDense(hidden_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_bins))

        # Create bin values (buffers are not parameters, but part of state_dict)
        bin_values = torch.linspace(min_v, max_v, num_bins).reshape(1, -1)
        self.register_buffer('bin_values', bin_values)

    def forward(self, x):
        value_feat = self.w1(x)
        value_feat = self.scaler(value_feat)
        # Logits
        logits = self.w2(value_feat) + self.bias

        # Log Probability
        log_prob = F.log_softmax(logits, dim=-1)
        
        # Expected Value: sum(prob * bin_value)
        # exp(log_prob) -> prob
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1, keepdim=True)

        info = {"log_prob": log_prob}
        return value, info

# -----------------------------------------
# Network Wrappers
# -----------------------------------------

class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,  # Added: PyTorch needs explicit input dim
        num_blocks: int,
        hidden_dim: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float
    ):
        super().__init__()
        
        self.embedder = HyperEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )
        
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
        
        self.predictor = HyperDeterministicPolicy(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations):
        y = self.embedder(observations)
        z = self.encoder(y)
        mean = self.predictor(z)
        return mean


class CategoricalCritic(nn.Module):
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
    ):
        super().__init__()
        
        self.embedder = HyperEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )
        
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

        self.predictor = HyperCategoricalValue(
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], dim=-1)
        y = self.embedder(x)
        z = self.encoder(y)
        q, info = self.predictor(z)
        return q, info

class CategoricalValue(nn.Module):
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
    ):
        super().__init__()
        
        self.embedder = HyperEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )
        
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

        self.predictor = HyperCategoricalValue(
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations,):
        x = observations
        y = self.embedder(x)
        z = self.encoder(y)
        value, info = self.predictor(z)
        return value, info



