### `muon_pkg`: SAC + Muon optimizer experiments

This folder contains a **self-contained SAC training pipeline** designed to test optimizer hypotheses:

- **Muon vs AdamW/Adam**
- **Momentum effects** (Muon momentum / Adam beta1), separately for actor and critic
- **Which layers** should use Muon (first/last layer toggles)
- **Delayed momentum** (“turn on” momentum later) via schedules

It reuses the repo’s `ReplayBuffer` (`data_buffer/replay_buffer.py`) unchanged.

### Files

- **`muon_pkg/muon.py`**: Muon optimizer implementation.
- **`muon_pkg/toy_model.py`**: MLP actor + double-Q critic with:
  - residual MLP option
  - activation norm options: `none | layernorm | l2`
  - spectral normalization option
- **`muon_pkg/optim_utils.py`**: optimizer factory (3 cases) + momentum schedules.
- **`muon_pkg/sac.py`**: SAC algorithm (`BaseAlgo`-compatible run loop + checkpointing).
- **`muon_pkg/train_dmc_sac.py`**: DMC entrypoint (Hydra).
- **`muon_pkg/configs/*.yaml`**: configs for SAC runs & sweeps.

### Run SAC (DMC)

From repo root:

```bash
python muon_pkg/train_dmc_sac.py
```

By default it loads `muon_pkg/configs/sac_cheetah.yaml`, which is set up as a **Muon (Case3) example**.

### Dependencies

This module expects the same core deps as the rest of the repo (at minimum): `torch`, `numpy`, `hydra-core`, `omegaconf`, `gymnasium`, and `dm_control`.

### Key config knobs (what to sweep)

All knobs live under `training.*` in `muon_pkg/configs/sac_dmc.yaml` and overrides in `sac_cheetah.yaml`.

#### Optimizer choice (hypothesis #1)

For each of **actor** and **critic**:

- **`training.actor_optimizer.case`** / **`training.critic_optimizer.case`**:
  - `"adam"`: all params use Adam
  - `"adamw"`: all params use AdamW
  - `"muon"`: Muon for most 2D weights + fallback Adam/AdamW for the rest (**recommended**)

Examples:

```bash
# AdamW baseline
python muon_pkg/train_dmc_sac.py \
  method_name=SAC_AdamW \
  training.actor_optimizer.case=adamw training.critic_optimizer.case=adamw

# Muon (Case3) on both actor and critic
python muon_pkg/train_dmc_sac.py \
  method_name=SAC_Muon \
  training.actor_optimizer.case=muon training.critic_optimizer.case=muon
```

#### Momentum effects (hypothesis #2)

- For **Muon**, momentum is `training.*_optimizer.muon.momentum`
- For **Adam/AdamW**, the “momentum-like” term is `beta1` inside `training.*_optimizer.betas`

You can sweep:

- **Muon momentum**: `training.critic_optimizer.momentum=0.0,0.9,0.95,0.99`
- **Adam beta1**: `training.critic_optimizer.betas=[0.0,0.999]` (keep beta2 fixed)

#### Layer selection: “which layers should use Muon?” (hypothesis #3)

We keep this deliberately simple (no regex):

- **`training.*_optimizer.muon.apply_first_layer`**
- **`training.*_optimizer.muon.apply_last_layer`**

Heuristics (based on `muon_pkg/toy_model.py`):

- Actor “last layer” = both `mean` and `log_std` heads
- Critic “last layer” = both `q1.out` and `q2.out` heads
- Critic “first layer” = both `q1.fcs[0]` and `q2.fcs[0]`

Example: Muon only on critic, keep actor pure AdamW (Case2):

```bash
python muon_pkg/train_dmc_sac.py \
  method_name=SAC_CriticMuonOnly \
  training.actor_optimizer.case=adamw \
  training.critic_optimizer.case=muon
```

Example: exclude last layer from Muon (e.g., actor Gaussian heads):

```bash
python muon_pkg/train_dmc_sac.py \
  method_name=SAC_ActorMuonNoHeads \
  training.actor_optimizer.case=muon \
  training.actor_optimizer.muon.apply_last_layer=false
```

#### Delayed momentum (“turn on later”) (hypothesis #4)

Schedules apply to:

- Muon: `param_group["momentum"]`
- Adam/AdamW: `param_group["betas"][0]` (beta1)

Configs:

- **`training.actor_momentum_schedule`**
- **`training.critic_momentum_schedule`**

Schedule behavior (simplified):

- `enabled: false`: no schedule (keep optimizer defaults)
- `enabled: true`: 0.0 until `start_step`, then use the optimizer default (Muon momentum / Adam beta1)

Example: turn on critic momentum/beta1 after 200k steps:

```bash
python muon_pkg/train_dmc_sac.py \
  method_name=SAC_MomentumDelayed \
  training.critic_momentum_schedule.enabled=true \
  training.critic_momentum_schedule.start_step=200000
```

### H3: Feature rank collapse metric (effective rank)

Enable periodic SVD diagnostics of the critic’s **penultimate features**:

```bash
python muon_pkg/train_dmc_sac.py \
  training.rank_logging.enabled=true \
  training.rank_logging.every_n_updates=1000 \
  training.rank_logging.max_samples=512
```

Logged metrics include `repr/q_effective_rank_mean`, per-Q effective ranks, and top-k singular-value mass.

### Sweeping with Hydra (multi-run)

Hydra built-in sweeps:

```bash
python muon_pkg/train_dmc_sac.py -m \
  training.actor_optimizer.case=adamw,muon \
  training.critic_optimizer.case=adamw,muon \
  training.actor_optimizer.muon.momentum=0.0,0.95 \
  training.critic_optimizer.muon.momentum=0.0,0.95
```

Tip: give each run a distinct `method_name` if you want separate checkpoint folders:

```bash
python muon_pkg/train_dmc_sac.py -m \
  method_name='SAC_${training.actor_optimizer.case}_${training.critic_optimizer.case}' \
  training.actor_optimizer.case=adamw,muon \
  training.critic_optimizer.case=adamw,muon
```

### Notes / gotchas

- **Muon requires CUDA** in this implementation (it allocates CUDA buffers).
- `case: "muon"` uses a fallback optimizer for 0/1D params by default; you usually want this.
- Hydra changes the working directory per run; if you want checkpoints in a stable location, set `training.checkpoint_dir` / `training.replay_dir` to absolute paths.


