# Installation Notes

This file documents the setup path used by the submission package. The professor-facing workflow is DMC-only and does not require the optional benchmark dependencies under `third_party/`.

## Canonical setup

```bash
conda env create -f environment.yml
conda activate oracles
bash setup_paths.sh
```

`environment.yml` is the canonical environment definition. `setup_paths.sh` writes `config/paths_local.yaml` so tracked configs do not need hard-coded machine paths.

## What is required for the submission path

Required:

- Python 3.10
- PyTorch
- Hydra / OmegaConf
- DeepMind Control (`dm-control`)
- test utilities such as `pytest`

Not required for the professor review path:

- `third_party/CARL`
- `third_party/Metaworld`
- `third_party/HighwayEnv`
- `third_party/myosuite`
- Box2D-specific extras

## Optional broader research dependencies

The repository still contains code for Metaworld, HighwayEnv, MyoSuite, and custom Box2D environments. Those workflows require additional dependencies and, in several cases, local unversioned clones under `third_party/`.

`install.sh` remains a stricter convenience wrapper for those broader workflows. It is not the primary setup path for the submission package.

## Validation commands

Recommended after setup:

```bash
python -m scripts.smoke_dmc --config-name dmc/submission_smoke_cartpole
python -m pytest tests -q
python -m compileall algo env model data_buffer scripts tests
```

## Known limitation

From a clean clone, the non-DMC benchmark families are not fully self-contained because some required external repositories are not tracked as submodules. This does not block the DMC-only submission package, but it does limit full-repo reproducibility.
