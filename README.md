# Oracle-Guided-RL

Oracle-guided off-policy reinforcement learning for continuous control in the **DeepMind Control (DMC)** suite. This project trains a single agent (the *learner*) to solve tasks such as Cheetah Run, Cartpole Swingup, and Walker Run, while using one or more **oracles**—pre-trained (and possibly suboptimal) policies—as guides. The learner is trained to improve beyond the oracles.

## Overview

The program supports three workflows:

1. **Prepare oracles** — `train_simba.py` runs standard off-policy actor-critic (SIMBA) training **without** oracles. It produces policy checkpoints that you then register as oracles. This is **not** a baseline; it is the tool to **prepare oracle checkpoints**.

2. **Our method: CurrimaxAdv** — `train_CurrimaxAdv_simba.py` trains the learner alongside multiple oracle policies. At each step, the learner and each oracle propose action candidates; critics score them to build an oracle-guided behavior policy for exploration. The learner is updated with oracle guidance and can surpass the oracles over time.

3. **Baselines** — Oracle-guided baselines for comparison: LOKI, CUP, MAPS, QMIX, MaxVQ (in `scripts/baselines/`). All use the same oracle checkpoints for fair comparison.

## Setup

**Requirements:** Conda (Miniconda or Anaconda)

From the project root:

```bash
bash install.sh
conda activate opi
```

This creates a Conda environment named `opi` and installs PyTorch, Hydra, DeepMind Control, Gymnasium, and other dependencies. For headless rendering (e.g. on a cluster), set:

```bash
export MUJOCO_GL=egl
```

### Path configuration

Oracle checkpoints use `${paths.project_root}` in configs. To set the project root:

- Run from the project directory (default), or  
- Set `export ORACLES_PROJECT_ROOT=/path/to/Oracle-Guided-RL`, or  
- Create `config/paths_local.yaml` with `project_root: /your/path` (see `config/paths_local.yaml.example`)

## Quick Start

### 1. Prepare oracles (required before CurrimaxAdv or baselines)

Train a policy and save checkpoints:

```bash
python -m scripts.train_simba --config-name dmc/simba_cheetah
```

Checkpoints are written under `checkpoints/`. Then point the oracle config at those paths (see [Oracle configuration](#oracle-configuration) below).

### 2. Run CurrimaxAdv (oracle-guided method)

```bash
python -m scripts.train_CurrimaxAdv_simba --config-name dmc/CurrimaxAdv_cheetah
```

Override from the command line:

```bash
python -m scripts.train_CurrimaxAdv_simba --config-name dmc/CurrimaxAdv_walker training.n_oracles=2 seed=0
```

### 3. Run baselines

Each baseline shares the same oracle config for a task to ensure fair comparison:

```bash
python -m scripts.baselines.train_loki_simba --config-name baselines_configs/loki/cheetah/loki_cheetah
python -m scripts.baselines.train_cup_simba --config-name baselines_configs/cup/cheetah/cup_cheetah
python -m scripts.baselines.train_maps_simba --config-name baselines_configs/maps/cheetah/maps_cheetah
python -m scripts.baselines.train_qmix_simba --config-name baselines_configs/qmix/cheetah/qmix_cheetah
python -m scripts.baselines.train_maxVQ_simba --config-name baselines_configs/maxVQ/cheetah/maxVQ_cheetah
```

## Oracle configuration

Oracle checkpoint paths are defined in task-specific oracle configs under `config/oracles/dmc/`, e.g. `cheetah_run.yaml`:

```yaml
oracles_dict:
  oracle_0: '${paths.project_root}/checkpoints/cheetah_run/Simba_Cheetah/seed42/checkpoints/Simba_Cheetah_seed42_best.pt'
  oracle_1: '${paths.project_root}/checkpoints/cheetah_run/Simba_Cheetah/seed42/checkpoints/Simba_Cheetah_seed42_75999.pt'
  oracle_2: null
```

- Set each `oracle_*` to a full path to a `.pt` checkpoint, or `null` if unused.  
- `training.n_oracles` must match the number of non-null oracles.  
- CurrimaxAdv and all baselines for a given task include the **same** oracle config (e.g. `- /oracles/dmc/cheetah_run`) in their Hydra `defaults`, so they use the same oracles.

## Configuration structure

All runs are configuration-driven via **Hydra**. Configs are composed from multiple YAML files:

| Directory | Purpose |
|-----------|---------|
| `config/base_configs/` | Algorithm defaults: `simba_dmc.yaml`, `maxAdv_dmc.yaml`, `loki_dmc.yaml`, `cup_dmc.yaml`, etc. |
| `config/dmc/` | Task-level run configs for SIMBA and CurrimaxAdv (e.g. `simba_cheetah.yaml`, `CurrimaxAdv_cheetah.yaml`) |
| `config/oracles/dmc/` | Oracle checkpoint paths per task (`oracles_dict`) |
| `config/baselines_configs/<method>/<task>/` | Baseline run configs (e.g. `loki/cheetah/loki_cheetah.yaml`) |

Each run config’s `defaults:` lists a base config, the task oracle config (for oracle-guided runs), and `_self_` for overrides. CurrimaxAdv hyperparameters live in `config/base_configs/maxAdv_dmc.yaml`; baseline hyperparameters in the corresponding base configs.

## Repository structure

| Path | Description |
|------|-------------|
| `algo/` | Training algorithms, replay logic, artifact helpers |
| `config/` | Hydra configs (dmc, base_configs, oracles, baselines_configs) |
| `data_buffer/` | Replay buffer |
| `env/` | Environment factories and wrappers |
| `model/` | Actor/critic networks (`simba.py`, `simba_base.py`) |
| `scripts/` | Entry points: `train_simba.py`, `train_CurrimaxAdv_simba.py` |
| `scripts/baselines/` | Baseline entry points (loki, cup, maps, qmix, maxVQ) |

## Outputs

- **Checkpoints** (`.pt`): `checkpoints/<task>/<method>/seed<seed>/checkpoints/`
- **Replay buffer**: `checkpoints/<task>/<method>/seed<seed>/replay/`
- **Run manifest** / config snapshot: `run.json`, `cfg.yaml`
- **Evaluation**: `eval/<task>/eval/<method>/`
- **Videos** (if enabled): `checkpoints/<task>/<method>/seed<seed>/videos/`

