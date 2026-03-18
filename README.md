# Oracle-Guided-RL

Oracle-Guided-RL is a research codebase for comparing oracle-guided reinforcement learning methods against strong baselines across DeepMind Control, Box2D-style hybrid environments, and Metaworld tasks. The repository keeps experiment logic config-driven through Hydra and stores benchmark variants as tracked YAML configs.

## Canonical Setup

Use the root Conda file as the canonical environment entry point:

```bash
git clone <repo-url>
cd Oracle-Guided-RL
conda env create -f environment.yml
conda activate oracles
bash setup_paths.sh
```

`setup_paths.sh` creates `config/paths_local.yaml`, which keeps machine-specific paths out of tracked configs.

### Benchmark-specific extras

- DeepMind Control quick start: no extra local clones beyond the Conda environment.
- Metaworld, CARL, HighwayEnv, and MyoSuite runs require local checkouts under `third_party/`.
- `install.sh` is a strict convenience wrapper for environments where those `third_party/` directories already exist.

## Quick Start

Minimal tracked smoke run:

```bash
python -m scripts.train_simba \
  --config-name dmc/simba_cartpole_sparse \
  training.total_timesteps=1000 \
  training.eval_every=500 \
  training.save_freq=500 \
  training.use_wandb=false \
  training.track=false \
  training.save_video=false \
  training.resume=false \
  training.use_compile=false \
  seed=0
```

Representative full training entry points:

```bash
python -m scripts.train_CurrimaxAdv --config-name dmc/CurrimaxAdv_cheetah seed=42
python -m scripts.train_bc --config-name dmc/bc_humanoid seed=42
python -m scripts.train_CurrimaxAdv_test --config-name metaworld/CurrimaxAdv_assembly seed=42
```

## Project Layout

- `algo/`: algorithm implementations
- `config/`: Hydra configs, grouped by benchmark family
- `env/`: environment builders and wrappers
- `model/`: policy and value networks
- `scripts/`: tracked training entry points
- `third_party/`: required local benchmark dependencies not versioned by the main repo
- `tests/`: lightweight smoke tests for config composition and model instantiation

## Hydra Usage

Configs are selected with `--config-name`, and overrides are passed as `key=value` pairs:

```bash
python -m scripts.train_simba \
  --config-name dmc/simba_hopper \
  training.total_timesteps=20000 \
  training.use_wandb=false \
  seed=7
```

The repository does not require ignored local launchers for reproduction; review-facing commands should use tracked Python entry points directly.

## Validation

Static validation:

```bash
python -m compileall algo env model data_buffer scripts muon_pkg viz tests
```

Test suite:

```bash
python -m pytest tests -q
```

## Reproducibility Notes

- Tracked experiment definitions live in `config/`; local paths should go only in `config/paths_local.yaml`.
- Large artifacts such as `checkpoints/`, `outputs/`, and `eval/` are intentionally not versioned.
- Current blocker: some benchmarks require `third_party/` clones that are not tracked by the main repository and are not submodules. The minimal structural fix is to add pinned submodules or a pinned bootstrap script. Until then, setup for those benchmarks is explicit but not fully self-contained from a clean clone.
- Visualization configs now use repository-relative paths; generated figures still remain local artifacts.
