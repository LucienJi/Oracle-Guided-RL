from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from muon_pkg.muon import Muon


def _named_trainable_params(module: nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in module.named_parameters() if p.requires_grad]


def _dedupe_params(params: Sequence[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    # Keep order, remove duplicates by identity.
    seen: set[int] = set()
    out: List[torch.nn.Parameter] = []
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _first_last_weight_params(module: nn.Module, *, two_d_named: Sequence[Tuple[str, torch.nn.Parameter]]) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Heuristic "first/last layer" detection tailored to `muon_pkg/toy_model.py`:
    - Actor: first = trunk.fcs[0].weight, last = mean.weight & log_std.weight
    - Critic: first = q1.fcs[0].weight & q2.fcs[0].weight, last = q1.out.weight & q2.out.weight
    Fallback: first/last among all 2D+ params by `named_parameters()` order.
    """
    first: List[torch.nn.Parameter] = []
    last: List[torch.nn.Parameter] = []

    # Actor-like (SquashedGaussianActor)
    if hasattr(module, "trunk") and hasattr(module, "mean") and hasattr(module, "log_std"):
        trunk = getattr(module, "trunk", None)
        if trunk is not None and hasattr(trunk, "fcs"):
            fcs = getattr(trunk, "fcs", None)
            if isinstance(fcs, (nn.ModuleList, list, tuple)) and len(fcs) > 0 and isinstance(fcs[0], nn.Linear):
                first.append(fcs[0].weight)
        mean = getattr(module, "mean", None)
        log_std = getattr(module, "log_std", None)
        if isinstance(mean, nn.Linear):
            last.append(mean.weight)
        if isinstance(log_std, nn.Linear):
            last.append(log_std.weight)
        if first or last:
            return _dedupe_params(first), _dedupe_params(last)

    # Critic-like (DoubleQCritic)
    if hasattr(module, "q1") and hasattr(module, "q2"):
        for q in (getattr(module, "q1", None), getattr(module, "q2", None)):
            if q is None:
                continue
            if hasattr(q, "fcs"):
                fcs = getattr(q, "fcs", None)
                if isinstance(fcs, (nn.ModuleList, list, tuple)) and len(fcs) > 0 and isinstance(fcs[0], nn.Linear):
                    first.append(fcs[0].weight)
            out = getattr(q, "out", None)
            if isinstance(out, nn.Linear):
                last.append(out.weight)
        if first or last:
            return _dedupe_params(first), _dedupe_params(last)

    # Fallback: just take the first/last 2D+ params in order
    two_d_params = [p for _, p in two_d_named]
    if two_d_params:
        first.append(two_d_params[0])
        last.append(two_d_params[-1])
    return _dedupe_params(first), _dedupe_params(last)


def _partition_params_for_case_muon(
    module: nn.Module,
    *,
    apply_first_layer: bool,
    apply_last_layer: bool,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Case3: Muon on most 2D+ params, with optional exclusion of "first layer" and/or "last layer".
    All 0/1D params always go to fallback.
    Returns: (muon_params, fallback_params)
    """
    named = _named_trainable_params(module)
    two_d_named = [(n, p) for n, p in named if p.ndim >= 2]
    other_params = [p for _, p in named if p.ndim < 2]

    first_params, last_params = _first_last_weight_params(module, two_d_named=two_d_named)

    excluded: List[torch.nn.Parameter] = []
    if not bool(apply_first_layer):
        excluded.extend(first_params)
    if not bool(apply_last_layer):
        excluded.extend(last_params)
    excluded_ids = {id(p) for p in excluded}

    muon_params = [p for _, p in two_d_named if id(p) not in excluded_ids]
    fallback_params = list(other_params) + [p for _, p in two_d_named if id(p) in excluded_ids]
    return _dedupe_params(muon_params), _dedupe_params(fallback_params)


@dataclass
class MomentumSchedule:
    """
    Momentum schedule controller for Muon (momentum) and Adam/AdamW (beta1).

    Simplified behavior:
      - If enabled=False: do nothing (keep optimizer's original momentum/beta1).
      - If enabled=True: set momentum/beta1 to 0.0 before start_step,
        then set it to a provided default (or an explicit value override) at/after start_step.

    Config schema (dict):
      enabled: bool
      step_unit: "global_step" | "update_step"
      start_step: int
      reset_state_on_start: bool
      value: Optional[float]  # if provided, overrides the default value
    """

    enabled: bool = False
    step_unit: str = "global_step"
    start_step: int = 0
    value: Optional[float] = None
    reset_state_on_start: bool = False

    _started: bool = False

    @staticmethod
    def from_cfg(cfg: Optional[Dict]) -> "MomentumSchedule":
        if not cfg:
            return MomentumSchedule(enabled=False)

        # Backward-compat: accept older schema with mode=("none"|"delayed").
        if "mode" in cfg:
            mode = str(cfg["mode"]).lower()
            if mode == "none":
                return MomentumSchedule(enabled=False, step_unit=str(cfg.get("step_unit", "global_step")))
            if mode == "delayed":
                return MomentumSchedule(
                    enabled=True,
                    step_unit=str(cfg.get("step_unit", "global_step")),
                    start_step=int(cfg.get("start_step", 0)),
                    value=float(cfg.get("value", 0.0)),
                    reset_state_on_start=bool(cfg.get("reset_state_on_start", False)),
                )
            raise ValueError(f"MomentumSchedule: unsupported legacy mode: {mode} (supported: none, delayed)")

        return MomentumSchedule(
            enabled=bool(cfg.get("enabled", False)),
            step_unit=str(cfg.get("step_unit", "global_step")),
            start_step=int(cfg.get("start_step", 0)),
            value=(None if cfg.get("value", None) is None else float(cfg.get("value"))),
            reset_state_on_start=bool(cfg.get("reset_state_on_start", False)),
        )

    def value_at(self, step: int, *, default_value: float) -> Optional[float]:
        step = int(step)
        if not self.enabled:
            return None
        target = float(self.value) if self.value is not None else float(default_value)
        return float(target if step >= self.start_step else 0.0)

    def maybe_apply(
        self,
        optimizers: Sequence[torch.optim.Optimizer],
        *,
        step: int,
        default_value: float,
    ) -> Optional[float]:
        """
        Applies the schedule at `step` by updating:
        - Muon: param_group['momentum']
        - Adam/AdamW: param_group['betas'] = (beta1, beta2)
        Returns the applied momentum/beta1 value (or None if schedule disabled).
        """
        v = self.value_at(step, default_value=float(default_value))
        if v is None:
            return None

        # Detect schedule start for optional state reset.
        schedule_started = step >= self.start_step
        do_reset = self.reset_state_on_start and schedule_started and (not self._started)

        for opt in optimizers:
            is_muon = isinstance(opt, Muon)
            if is_muon:
                for pg in opt.param_groups:
                    pg["momentum"] = float(v)
            else:
                for pg in opt.param_groups:
                    if "betas" in pg:
                        b1, b2 = pg["betas"]
                        pg["betas"] = (float(v), float(b2))

            if do_reset:
                _reset_optimizer_momentum_state(opt)

        if schedule_started:
            self._started = True

        return float(v)


def _reset_optimizer_momentum_state(opt: torch.optim.Optimizer) -> None:
    """
    Reset the *momentum-like* state in common optimizers.
    - Muon: state[p]['momentum_buffer']
    - Adam/AdamW: state[p]['exp_avg'] (keep exp_avg_sq to preserve variance estimate)
    """
    with torch.no_grad():
        for pg in opt.param_groups:
            for p in pg["params"]:
                if p not in opt.state:
                    continue
                st = opt.state[p]
                if "momentum_buffer" in st and torch.is_tensor(st["momentum_buffer"]):
                    st["momentum_buffer"].zero_()
                if "exp_avg" in st and torch.is_tensor(st["exp_avg"]):
                    st["exp_avg"].zero_()


def build_optimizer_stack(
    module: nn.Module,
    *,
    cfg: Dict,
    role: str,
    rank: int = 0,
    world_size: int = 1,
) -> List[torch.optim.Optimizer]:
    """
    Build one or more optimizers for `module` according to cfg.

    Experimental, simplified cases (no regex parameter selection):
      - Case1: all params use Adam
      - Case2: all params use AdamW
      - Case3: Muon for most 2D+ params, optionally excluding first/last layer weights;
               all remaining params use fallback Adam/AdamW.

    Config schema:
      cfg.case: "adam" | "adamw" | "muon"
      If case in ("adam","adamw"):
        lr, weight_decay, betas, eps
      If case == "muon":
        muon: {lr, weight_decay, momentum, nesterov, ns_steps, lowrank, apply_first_layer, apply_last_layer}
        fallback: {type: "adam"|"adamw", lr, weight_decay, betas, eps}
    """
    case = str(cfg["case"]).lower()

    if case in ("adam", "adamw"):
        params = [p for _, p in _named_trainable_params(module)]
        if not params:
            raise ValueError(f"[{role}] No trainable parameters.")
        lr = float(cfg["lr"])
        weight_decay = float(cfg["weight_decay"])
        betas = tuple(cfg["betas"])
        eps = float(cfg["eps"])
        if case == "adam":
            return [torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=True)]
        return [torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=True)]

    if case == "muon":
        if not torch.cuda.is_available():
            raise RuntimeError(f"[{role}] Muon optimizer requires CUDA (Muon allocates CUDA update buffers).")

        mu = dict(cfg["muon"])
        fb = dict(cfg["fallback"])

        muon_params, fallback_params = _partition_params_for_case_muon(
            module,
            apply_first_layer=bool(mu["apply_first_layer"]),
            apply_last_layer=bool(mu["apply_last_layer"]),
        )

        opts: List[torch.optim.Optimizer] = []
        if muon_params:
            opts.append(
                Muon(
                    muon_params,
                    lr=float(mu["lr"]),
                    weight_decay=float(mu["weight_decay"]),
                    momentum=float(mu["momentum"]),
                    nesterov=bool(mu["nesterov"]),
                    ns_steps=int(mu["ns_steps"]),
                    rank=int(rank),
                    world_size=int(world_size),
                    lowrank=bool(mu["lowrank"]),
                )
            )

        if fallback_params:
            fb_type = str(fb["type"]).lower()
            fb_lr = float(fb["lr"])
            fb_wd = float(fb["weight_decay"])
            fb_betas = tuple(fb["betas"])
            fb_eps = float(fb["eps"])
            if fb_type == "adam":
                opts.append(torch.optim.Adam(fallback_params, lr=fb_lr, weight_decay=fb_wd, betas=fb_betas, eps=fb_eps, foreach=True))
            elif fb_type == "adamw":
                opts.append(torch.optim.AdamW(fallback_params, lr=fb_lr, weight_decay=fb_wd, betas=fb_betas, eps=fb_eps, foreach=True))
            else:
                raise ValueError(f"[{role}] fallback.type must be 'adam' or 'adamw', got: {fb_type}")

        if not opts:
            raise ValueError(f"[{role}] case='muon' produced no optimizers (no params matched).")
        return opts

    raise ValueError(f"[{role}] Unknown optimizer case: {case}")


