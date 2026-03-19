# Report Draft: Oracle-Guided-RL DMC Submission Package

## 1. What the program does

This program is a Python reinforcement learning training package for DeepMind Control (DMC) tasks. In the professor-facing submission scope, it supports two related workflows:

1. a SIMBA baseline trainer (`scripts/train_simba.py`), and
2. an oracle-guided trainer (`scripts/train_CurrimaxAdv.py`).

Both workflows are driven by Hydra configuration files under `config/dmc/`. The program constructs a DMC environment, creates neural network policies and value models, manages replay and checkpoint artifacts, and runs training and evaluation loops. The code is intended primarily for experiment execution on a computation node or through Slurm, but it also includes a lightweight local smoke-test path so a reviewer can verify the basic setup and control flow without launching a full training job.

The repository contains additional research code for other benchmark families, including Box2D, Metaworld, Highway, and MyoSuite. Those workflows are not the submission focus because they require optional or unversioned dependencies. The submission package therefore emphasizes the DMC path, which is the cleanest self-contained programming project inside the repository.

## 2. How the program works

### Overall pipeline

The execution model is configuration-driven. A typical run proceeds as follows:

1. Hydra composes a configuration from a tracked YAML file and any command-line overrides.
2. The training script validates the key fields needed for the DMC run.
3. The code builds artifact directories and writes a resolved run manifest.
4. The environment factory in `env/env_utils.py` creates a DMC environment and a matching evaluation environment.
5. The selected training script constructs the actor, critic, and (for the oracle-guided path) value/oracle models.
6. The algorithm class in `algo/` runs the training loop, sampling actions, collecting transitions, updating networks, evaluating periodically, and writing checkpoints.

### Key modules

- `scripts/train_simba.py`
  This is the main entry point for the SIMBA baseline. It validates the configuration, creates the DMC environment, constructs the policy and critic, initializes the replay buffer, writes the config snapshot, and launches the training loop.

- `scripts/train_CurrimaxAdv.py`
  This is the main entry point for the oracle-guided variant. In addition to the learner models, it constructs oracle actors/critics from configuration and then launches the CurrimaxAdv training loop.

- `scripts/smoke_dmc.py`
  This is a lightweight submission helper added for review. It does not run a full experiment. Instead, it composes a DMC config, builds the environment, instantiates the model path, takes a few inference steps, and prints a small JSON summary. Its purpose is to provide a minimal local verification path.

- `env/env_utils.py`
  This module contains the environment wrappers and factory functions. One important submission-oriented cleanup was to defer optional benchmark imports until their corresponding factories are used. As a result, the DMC submission path can now be imported without also requiring Metaworld, HighwayEnv, MyoSuite, or custom Box2D packages.

- `algo/artifacts.py`
  This module centralizes output directory construction. It keeps runs organized by task, method, and seed, and writes a `run.json` manifest containing the resolved configuration and output paths.

- `data_buffer/replay_buffer.py`
  This module implements the replay buffer used during training. It stores transitions in memory and can also persist episodes to disk.

### Algorithmic components

The SIMBA path uses deterministic actor and critic networks defined in `model/simba.py`. The oracle-guided path reuses the same general modeling components but adds multiple oracle policies and supporting critics/value functions. The training scripts do not implement the update logic directly; instead, they instantiate algorithm classes from `algo/`, which contain the actual rollout, update, evaluation, and checkpointing logic.

## 3. How to compile / set up the program

Because this is a Python project, “compilation” means environment creation and dependency installation rather than producing a binary executable.

The required setup steps for the submission package are:

```bash
conda env create -f environment.yml
conda activate oracles
bash setup_paths.sh
```

`environment.yml` is the canonical environment definition. It installs the Python runtime, PyTorch, Hydra, DeepMind Control, and the basic test/developer tools. `setup_paths.sh` writes `config/paths_local.yaml`, which stores the local repository root path without modifying tracked configs.

This setup is sufficient for the DMC submission package. The broader research repository contains other benchmark families that require extra optional dependencies, but those are not necessary for the review path described in this report.

## 4. How to run and use the program

### Minimal local review path

The smallest reproducible path is the smoke test:

```bash
python -m scripts.smoke_dmc --config-name dmc/submission_smoke_cartpole
```

This command verifies that:

- configuration composition works,
- the DMC environment can be created,
- the SIMBA actor/critic path can be instantiated,
- artifact directories can be prepared, and
- a few environment interaction steps can be executed.

### Short local training path

For a small end-to-end training example, the following command runs the SIMBA entry point with the lightweight submission config:

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

### Oracle-guided path

The oracle-guided smoke configuration is:

```bash
python -m scripts.smoke_dmc --config-name dmc/submission_currimaxadv_smoke
```

This configuration uses `null` oracle checkpoints so the oracle-guided code path can be built without requiring pre-existing oracle files.

### Slurm / cluster path

The realistic full-experiment path remains cluster-oriented. A representative batch script is provided at `slurm/dmc_train_simba.sbatch`. After editing its placeholders for account, partition, and Conda initialization, it can be submitted with:

```bash
sbatch slurm/dmc_train_simba.sbatch
```

This preserves the project’s real usage model instead of pretending the repository is primarily a desktop application.

## 5. Input and output formats

### Inputs

The main inputs are Hydra configuration files and command-line overrides.

Examples:

- `--config-name dmc/submission_smoke_cartpole`
- `seed=1`
- `training.total_timesteps=5000`

The DMC configs specify:

- task selection (`domain_name`, `task_name`)
- observation layout (`obs_config`)
- training hyperparameters
- artifact and evaluation behavior

### Outputs

The program writes organized artifacts under the checkpoint/evaluation directory structure managed by `algo/artifacts.py`. Typical outputs include:

- `cfg.yaml`: resolved config snapshot
- `run.json`: run manifest
- replay data directory
- checkpoint `.pt` files during longer runs
- evaluation CSV files
- optional videos if video saving is enabled

## 6. Sample commands and expected outputs

### Sample command 1: smoke test

```bash
python -m scripts.smoke_dmc --config-name dmc/submission_smoke_cartpole
```

Expected output:

- JSON summary printed to stdout
- artifact directories created under `checkpoints/...`

### Sample command 2: short training run

```bash
python -m scripts.train_simba --config-name dmc/submission_smoke_cartpole training.total_timesteps=1000
```

Expected output:

- console logs showing parameter count and training start/completion
- `cfg.yaml` and `run.json`
- checkpoint and replay directories

## 7. Limitations

This submission does not claim that every benchmark family in the repository is locally reproducible. The DMC path is the clean submission unit because it avoids the optional external repositories needed by other parts of the research codebase.

In the current shell used for packaging, validation was limited because the project runtime environment was not active; for example, `hydra` and `pytest` were unavailable when command validation was attempted. As a result, static compilation and shell-script validation were completed, but full local runtime verification was not possible in this environment. This is documented explicitly in the submission checklist.
