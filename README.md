# Oracle-Guided-RL Submission Package

This submission packages a DeepMind Control (DMC) reinforcement learning training pipeline implemented in Python with Hydra-based configuration. The professor-facing review path focuses on two tracked entry points: `scripts/train_simba.py` for the SIMBA baseline and `scripts/train_CurrimaxAdv.py` for the oracle-guided variant. The repository still contains broader research code, but Box2D, Metaworld, Highway, and MyoSuite workflows are intentionally out of scope for the submission package because they rely on optional or unversioned dependencies.

## Submission Scope

- Primary workflow: DMC training with `python -m scripts.train_simba`
- Secondary workflow: DMC oracle-guided training with `python -m scripts.train_CurrimaxAdv`
- Review path: local smoke validation plus an honest Slurm/cluster execution path
- Out of scope for review: optional benchmark families that need `third_party/` clones or extra packages

## Repository Structure

- `algo/`: training algorithms and replay/artifact helpers
- `config/dmc/`: DMC experiment definitions, including submission smoke configs
- `env/`: environment factories and wrappers
- `model/`: actor/critic network implementations
- `scripts/`: runnable entry points and the `smoke_dmc.py` review helper
- `slurm/`: representative batch script for cluster execution
- `tests/`: lightweight composition/import tests for the submission package
- `examples/`: sample commands for the professor review path
- `report_draft.md`: short report draft required by the assignment
- `SUBMISSION_CHECKLIST.md`: explicit checklist of included deliverables and verified items

## Compilation / Setup

For this Python project, “compilation” means preparing the execution environment.

```bash
conda env create -f environment.yml
conda activate oracles
bash setup_paths.sh
```

`setup_paths.sh` writes `config/paths_local.yaml` with the current repository root. That file is gitignored and keeps machine-specific paths out of tracked configs.

### What this setup installs

- Core Python runtime and PyTorch
- Hydra and OmegaConf
- DeepMind Control dependencies for the DMC submission path
- Developer/test packages such as `pytest`

### Optional dependencies not required for the submission path

Metaworld, HighwayEnv, MyoSuite, CARL, and the custom Box2D benchmarks remain optional. They are not needed to review the DMC submission package.

## Quick Review Path

### 1. Local smoke test

```bash
python -m scripts.smoke_dmc --config-name dmc/submission_smoke_cartpole
```

What it does:

- composes the submission config,
- builds the DMC environment,
- instantiates the SIMBA actor/critic path,
- runs a few inference steps,
- prints a JSON summary including the artifact directories.

### 2. Short local training run

```bash
python -m scripts.train_simba \
  --config-name dmc/submission_smoke_cartpole \
  training.total_timesteps=1000 \
  training.eval_every=250 \
  training.save_freq=250 \
  training.use_wandb=false \
  training.save_video=false \
  training.resume=false \
  training.use_compile=false
```

### 3. Optional oracle-guided smoke/training path

```bash
python -m scripts.smoke_dmc --config-name dmc/submission_currimaxadv_smoke
python -m scripts.train_CurrimaxAdv --config-name dmc/submission_currimaxadv_smoke
```

The `submission_currimaxadv_smoke` config uses `null` oracle checkpoints so the oracle-guided path can be constructed without relying on bundled checkpoints.

## Slurm / Cluster Usage

The project is realistically intended for cluster execution when training beyond the smoke path.

Representative batch script:

```bash
sbatch slurm/dmc_train_simba.sbatch
```

The script includes placeholders for:

- partition
- account
- GPU count
- CPU count
- memory
- wall-clock time
- Conda initialization path

Typical monitoring commands:

```bash
squeue -u "$USER"
sacct -j <job_id> --format=JobID,JobName,State,Elapsed
```

## Inputs and Outputs

### Inputs

- Hydra config selected with `--config-name`
- optional Hydra overrides such as `seed=1` or `training.total_timesteps=5000`
- local path configuration written by `setup_paths.sh`

### Outputs

Artifact locations are generated automatically by `algo/artifacts.py`. For the submission configs, the main outputs are:

- `checkpoints/<task>/<method>/seed<seed>/checkpoints/`
- `checkpoints/<task>/<method>/seed<seed>/replay/`
- `checkpoints/<task>/<method>/seed<seed>/videos/` when video saving is enabled
- `checkpoints/<task>/<method>/seed<seed>/run.json`
- `eval/<task>/eval/<method>/`

## Example Commands

Additional ready-to-copy commands are listed in [examples/README.md](/share/data/ripl/jjt/projects/oracles/examples/README.md).

## Expected Artifacts

After a successful smoke or training run, expect at least:

- a resolved config snapshot `cfg.yaml`
- a run manifest `run.json`
- checkpoint and replay directories created under `checkpoints/`

Longer training runs may also produce:

- `.pt` checkpoint files
- evaluation CSV files
- videos if video saving is enabled

## Troubleshooting

- `ModuleNotFoundError: No module named 'torch'`:
  activate the Conda environment created from `environment.yml`.
- `ImportError` mentioning `metaworld`, `highway_env`, `myosuite`, or Box2D:
  you are trying to use an optional benchmark path that is not part of the DMC submission package.
- `ImportError` mentioning `dm_control.suite`:
  the DMC dependencies were not installed correctly.
- Hydra config/path issues:
  rerun `bash setup_paths.sh` and confirm `config/paths_local.yaml` was created.

## Limitations and Honest Caveats

- Full training is cluster-oriented; the guaranteed professor review path is the smoke test plus the documented Slurm script.
- The current submission package does not claim local reproducibility for every benchmark family in the repository.
- The repository still contains broader research code and ignored/generated artifact directories; the professor-facing path is intentionally narrower than the full research workspace.
- Validation in the current shell was limited by missing runtime packages; see `SUBMISSION_CHECKLIST.md` for verified versus unverified items.
