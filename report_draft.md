# Report Draft: Oracle-Guided-RL DMC Submission Package

## 1. What the program does

This program implements **oracle-guided off-policy reinforcement learning** for continuous control. The goal is to train a single agent (the *learner*) to solve tasks in the **DeepMind Control (DMC)** suite—e.g. Cheetah Run, Cartpole Swingup—while using one or more **oracles** as guides. Oracles are pre-trained (and possibly suboptimal) policies that suggest actions; the learner is trained to improve beyond them.

The program supports the following workflows:

1. **Prepare oracles: `train_simba.py`.** This script runs standard off-policy actor-critic (SIMBA) training **without** oracles—a single learner policy and critic. It is **not** a baseline for comparison; it is used to **prepare oracle checkpoints**. You train a policy, save `.pt` files under `checkpoints/`, then point the oracle config at those files so that the oracle-guided method and the baselines can load the same oracles.

2. **Our method: CurrimaxAdv** (`scripts/train_CurrimaxAdv_simba.py`). The learner is trained alongside multiple oracle policies. At each step, the learner and each oracle propose several action candidates. A set of critics (one for the learner and one per oracle) scores these candidates; we build an oracle-guided behavior policy for exploration. Transitions are stored in a replay buffer and used to update the learner’s actor and critic, as well as the oracle critics. During update, the learner is also updated with oracle guidance to accelerate training. Over time, the learner can surpass the oracles.

3. **Baselines (part of the project).** The repository implements several **oracle-guided baselines** for comparison: LOKI, CUP, MAPS, QMIX, MaxVQ, etc. (in `scripts/baselines/`). Each is an alternative algorithm that also consumes the same oracle checkpoints. For fair comparison, our method and all baselines for a given task use the **same** oracle config file (and thus the same `oracles_dict` checkpoint paths); see Section 2 and Section 4 for how to set this.



## 2. How the program works

### Overall pipeline

1. **Set the config in YAML.** All runs are configuration-driven. You choose a Hydra config (e.g. `config/dmc/CurrimaxAdv_cheetah.yaml`) and optionally override fields from the command line. Configs specify the task (domain/task name), observation layout, training hyperparameters, method name, and (for oracle-guided runs) the oracle checkpoint paths.

2. **Prepare oracles: `train_simba.py`.** This script is **not** a baseline—it is the tool to **prepare oracles**. It runs standard off-policy actor-critic with no oracles, trains a policy, and saves checkpoints under `checkpoints/`. Those `.pt` files are then referenced in the **oracle config** (see below) so that our method and all baselines load the same pre-trained policies as oracles.

3. **Our method: `train_CurrimaxAdv_simba.py`.** The oracle-guided algorithm. It loads oracle checkpoints from the config, builds learner and oracle actors/critics/values, and runs the CurrimaxAdv training loop with oracle-guided exploration and updates.

4. **Baselines (part of the project).** The repository implements oracle-guided **baselines** for comparison (e.g. in `scripts/baselines/`: LOKI, CUP, MAPS, QMIX, MaxVQ). Each has its own config and entry point. To ensure **fair comparison**, our method and every baseline for a given task must use the **same oracles**. This is done by having them all include the **same task-level oracle config** in their Hydra `defaults`. For example, for the Cheetah Run task, both `config/dmc/CurrimaxAdv_cheetah.yaml` and `config/baselines_configs/loki/cheetah/loki_cheetah.yaml` include `- /oracles/dmc/cheetah_run` in `defaults`. That file defines `oracles_dict` (paths to `.pt` checkpoints). As long as CurrimaxAdv and all baselines for that task use the same oracle config (e.g. `cheetah_run.yaml`), they all read the same `oracles_dict` and thus use the same set of oracle checkpoints. When preparing oracles, train with `train_simba`, save checkpoints, then set those paths in **one** place—`config/oracles/dmc/<task>.yaml` (e.g. `cheetah_run.yaml`)—and use that config for both our method and every baseline for that task.

5. **Training and evaluation outputs.** Artifact paths are managed by `algo/artifacts.py`. **Model checkpoints** (`.pt` files) are saved under the **`checkpoints/`** directory, organized by task, method name, and seed. **Evaluation results** (e.g. CSV files) are written under the **`eval/`** folder (e.g. `eval/<task_name>/eval/<method_name>/`). The run manifest `run.json` and resolved `cfg.yaml` are written in the same run directory.

6. **Extension: new tasks.** To add a new control task, extend the environment factory in **`env/env_utils.py`**: implement a wrapper or factory that builds the new environment and matches the expected observation/action interface. Once registered and wired into the config (e.g. a new DMC task or a new benchmark family), the same training scripts and artifact layout can be used for that task.

### Key modules

- `scripts/train_simba.py`
  Entry point for **preparing oracles** (not a baseline). It validates the configuration, creates the DMC environment, constructs the policy and critic, initializes the replay buffer, writes the config snapshot, and runs the SIMBA training loop. Checkpoints saved here are used as oracles by our method and baselines.

- `scripts/train_CurrimaxAdv_simba.py`
  Entry point for our oracle-guided method. It builds learner and oracle actors/critics/values from config (including oracle checkpoints), then runs the CurrimaxAdv training loop.

- `scripts/baselines/train_*_simba.py`
  Entry points for oracle-guided baselines (e.g. `train_loki_simba.py`, `train_cup_simba.py`, `train_maps_simba.py`). Each loads the same `oracles_dict` from the task’s oracle config and runs the corresponding algorithm.

- `model/simba.py` and `model/simba_base.py`
  **Model definitions.** The policy and critic used by SIMBA (and by oracles) are defined here. `simba_base.py` provides the base building blocks: `CategoricalCritic`, `CategoricalValue`, `DeterministicPolicy`, and the truncated-normal sampling used for exploration. `simba.py` defines `DeterministicSimbaPolicy` (actor), `SimbaCritics` (ensemble of categorical Q-networks with target networks), and `SimbaValues` (ensemble of categorical V-networks). Architecture and sizes are controlled by config (`rl_agent_args`, `rl_critic_args`, `simba_based_oracle_args`, etc.). Oracle policies loaded from checkpoints use the same SIMBA architecture.

- `env/env_utils.py`
  Environment wrappers and factory functions. Optional benchmark imports are deferred until their factories are used, so the DMC path can run without Metaworld, HighwayEnv, MyoSuite, or custom Box2D packages.

- `algo/artifacts.py`
  Centralizes output directory construction: runs organized by task, method, and seed; writes `run.json` and resolved config paths.

- `data_buffer/replay_buffer.py`
  Replay buffer for training: stores transitions in memory and can persist episodes to disk.


## 3. How to compile / set up the program

Because this is a Python project, “compilation” means environment creation and dependency installation rather than producing a binary executable.

**Environment requirement: Conda (Miniconda or Anaconda).** The project expects a Conda installation to create and activate the runtime environment.

**Default setup:** Run the install script from the project root:

```bash
bash install.sh
```

This creates a **new Conda environment named `opi`** (defined in `environment.yml`) and installs dependencies. To use a different environment name, edit the `name:` field at the top of `environment.yml` before running `conda env create -f environment.yml`, and adjust any `conda activate` or `conda run -n <name>` commands accordingly.

**What gets installed:** The canonical definition is `environment.yml`. It installs the Python runtime, **PyTorch** (and CUDA if available), Hydra, OmegaConf, DeepMind Control, Gymnasium, and the packages listed in `requirements/base.txt`. The main code dependency is **PyTorch**; the training code also uses **`torch.compile`** for parts of the update step. The **first run** may take longer while TorchDynamo compiles these paths; subsequent runs reuse the cache.

After installation, activate the environment and (if needed) set the project root for config paths:

```bash
conda activate opi
```

### Configuration (Hydra) and the `config/` directory

All run settings are maintained under **`config/`** using **Hydra**. Configs have **non-trivial dependency relationships**: the final run configuration is composed by merging several YAML files according to the `defaults` list and override order. Understanding this structure is necessary to change tasks, oracles, or hyperparameters correctly.

**How Hydra composes configs.** Each run is started with a **root config** (e.g. `dmc/CurrimaxAdv_cheetah` or `baselines_configs/loki/cheetah/loki_cheetah`). The root file’s `defaults:` list specifies which other configs to load and in what order; later entries override earlier ones for overlapping keys. The special entry `_self_` means “merge this file’s contents after the listed defaults,” so the root file can override anything brought in by the defaults. Hydra resolves `${...}` interpolations (e.g. `${run_name}`, `${paths.project_root}`) after composition. Optional overrides (e.g. `config/paths_local.yaml`) are typically included via Hydra’s search path or a `defaults` entry so that machine-specific paths do not live in tracked YAML.

**Main directories under `config/`:**


- **`config/base_configs/`**  
  **Base (algorithm) configs** default parameters for each algorithm shared across tasks. Examples: `simba_dmc.yaml` (for SIMBA algorithm's default parameter for DMC task), `maxAdv_dmc.yaml` (for CurrimaxAdv), `loki_dmc.yaml`, `cup_dmc.yaml`, `maps_dmc.yaml`, `qmix_dmc.yaml`, `maxVQ_dmc.yaml` (for baselines). They define training hyperparameters, optimizer settings, network args (`rl_agent_args`, `rl_critic_args`, etc.), and often include `defaults: - /paths` and oracle-related bases. A task config’s `defaults:` typically lists one of these first, so the task inherits the full algorithm setup and then overrides only what is task-specific.


- **`config/dmc/`**  
  Task-level **run configs** for DMC. Examples: `simba_cheetah.yaml`, `CurrimaxAdv_cheetah.yaml`. Each specifies `defaults:` (e.g. a base config + an oracle config), `env.domain_name`, `env.task_name`, `method_name`, `obs_config`, and optional `training:` overrides. These are the configs you pass as `--config-name dmc/...` when running the scripts.


- **`config/oracles/`**  
  **Oracle-related config.**  
  - **`config/oracles/dmc/<task>.yaml`** (e.g. `cheetah_run.yaml`, `walker_run.yaml`): Define **`oracles_dict`**—the mapping from names like `oracle_0`, `oracle_1` to checkpoint paths (`${paths.project_root}/checkpoints/.../file.pt`) or `null`. This is the **single place** to set which checkpoints are used as oracles for that task. For fair comparison, **our method and every baseline** for the same task should include the **same** file here in their `defaults` (e.g. `- /oracles/dmc/cheetah_run`), so they all use the same oracles.  
  - **`config/oracles/simba_based_oracle_args.yaml`**: Shared architecture args for SIMBA-based oracle policies (included by base configs that use oracles).  
  - Other files under `oracles/` may define MLP-based oracle args or shared oracle settings.


- **`config/baselines_configs/`**  
  **Baseline-specific run configs**, one per (method, task) or (method, variant). Structure is like `config/baselines_configs/<method>/<task>/<method>_<task>.yaml` (e.g. `loki/cheetah/loki_cheetah.yaml`). Each has `defaults:` that include the corresponding base config (e.g. `loki_dmc`) and the **same** task oracle config as used for CurrimaxAdv (e.g. `oracles/dmc/cheetah_run`), plus `_self_` for task/method overrides.

**Typical dependency chain for one run.** For `train_CurrimaxAdv_simba` with default config: Hydra loads `dmc/CurrimaxAdv_cheetah.yaml` → its `defaults` load `base_configs/maxAdv_dmc.yaml` and `oracles/dmc/cheetah_run`, then `_self_`. So algorithm hyperparameters and oracle paths come from those included files; the root file only overrides `method_name`, `env`, `obs_config`, and a few `training` keys. For a baseline, e.g. LOKI on Cheetah: root is `baselines_configs/loki/cheetah/loki_cheetah.yaml` → defaults load `base_configs/loki_dmc` and `oracles/dmc/cheetah_run`, then `_self_`. Again, `cheetah_run` supplies the same `oracles_dict` for fair comparison. When you add or change a task, you add or edit a run config in `dmc/` for our main algorithm or `baselines_configs/` for baselines, and a shared oracle config in `oracles/dmc/<task>.yaml` if oracles are used; base configs are reused across tasks.

## 4. How to run and use the program

### train_simba (prepare oracles — not a baseline)

**Purpose:** `train_simba` is **not** a baseline for comparison. It is the script used to **prepare oracle checkpoints**. You run it to train a policy without oracles, save `.pt` checkpoints, then point the oracle config (e.g. `config/oracles/dmc/cheetah_run.yaml`) at those paths so that our method and all baselines use the same oracles for that task.

**Command (default config):**

```bash
conda activate opi
python -m scripts.train_simba
```

The script uses the default config `dmc/simba_cheetah`. To use another task or override parameters:

```bash
python -m scripts.train_simba --config-name dmc/simba_walker
python -m scripts.train_simba training.total_timesteps=50000 training.eval_every=5000 seed=1
```

**Where hyperparameters come from:** The selected config (e.g. `config/dmc/simba_cheetah.yaml`) sets `env.domain_name`, `env.task_name`, `method_name`, `seed`, `run_name`. Base training defaults (e.g. `training.total_timesteps`, `batch_size`, `rl_agent_args`, `rl_critic_args`) come from `config/base_configs/simba_dmc.yaml`. Override via the task YAML, the base config, or the command line (e.g. `training.total_timesteps=100000`). Checkpoints are written under `checkpoints/`; set those paths in `config/oracles/dmc/<task>.yaml` as `oracles_dict` for our method and baselines.

### train_CurrimaxAdv_simba (oracle-guided method)

**Command (default config):**

```bash
conda activate opi
python -m scripts.train_CurrimaxAdv_simba
```

The default config is `dmc/CurrimaxAdv_cheetah`. To choose another task or override:

```bash
python -m scripts.train_CurrimaxAdv_simba --config-name dmc/CurrimaxAdv_walker
python -m scripts.train_CurrimaxAdv_simba training.n_oracles=2 seed=0
```

**Where to set oracles:** Oracle checkpoint paths are **not** in the main DMC config file; they are in the **oracle config** included via `defaults`. For `CurrimaxAdv_cheetah`, the defaults are:

- `config/base_configs/maxAdv_dmc.yaml` (algorithm base)
- `config/oracles/dmc/cheetah_run.yaml` (oracle paths for this task)

In **`config/oracles/dmc/cheetah_run.yaml`** you will find `oracles_dict`, for example:

```yaml
oracles_dict:
  oracle_0: '${paths.project_root}/checkpoints/cheetah_run/Simba_Cheetah/seed42/checkpoints/Simba_Cheetah_seed42_best.pt'
  oracle_1: '${paths.project_root}/checkpoints/cheetah_run/Simba_Cheetah/seed42/checkpoints/Simba_Cheetah_seed42_75999.pt'
  oracle_2: null
```

- Set each `oracle_*` to a full path to a `.pt` checkpoint (use `${paths.project_root}/...` so it works across machines), or to `null` if that slot is unused.
- `training.n_oracles` in the task config (e.g. `CurrimaxAdv_cheetah.yaml`) must match the number of non-null oracles you use. Other task-specific oracle configs live under `config/oracles/dmc/<task>.yaml` (e.g. `walker_run.yaml`, `quadruped_run.yaml`).

**Where to set algorithm parameters:** CurrimaxAdv-specific hyperparameters are in **`config/base_configs/maxAdv_dmc.yaml`**: e.g. `learner_learning_starts`, `per_oracle_explore_steps`, `num_proposals`, `kappa`, `oracle_std_multiplier`, `sampling_max_temperature` / `sampling_min_temperature`, `beta`, `l1_loss_weight`, `rl_loss_weight`, `baseline_greedy_guided`, and optimizer/logging settings. The task-level config (e.g. `config/dmc/CurrimaxAdv_cheetah.yaml`) overrides a few of these under `training:` (e.g. `n_oracles`, `baseline_greedy_guided`, `use_wandb`, `use_compile`). Change them in those YAML files or via CLI overrides like `training.num_proposals=3`.

### Baselines (LOKI, CUP, MAPS, QMIX, MaxVQ, etc.)

Baselines are oracle-guided methods used for comparison with CurrimaxAdv. Each has its own script under `scripts/baselines/` and configs under `config/baselines_configs/<method>/<task>/`. To run a baseline, activate the env and call the corresponding module with the desired config. The default `config_name` in each script points to one (method, task) pair; you can override it to switch task or pass CLI overrides.

**Examples:**

```bash
conda activate opi
# LOKI (default config in script may be e.g. loki/quad/loki_quad; override to run Cheetah)
python -m scripts.baselines.train_loki_simba --config-name baselines_configs/loki/cheetah/loki_cheetah

# CUP, MAPS, QMIX, MaxVQ: same idea — use config_name that includes the same oracle config for the task
python -m scripts.baselines.train_cup_simba --config-name baselines_configs/cup/cheetah/cup_cheetah
python -m scripts.baselines.train_maps_simba --config-name baselines_configs/maps/cheetah/maps_cheetah
python -m scripts.baselines.train_qmix_simba --config-name baselines_configs/qmix/cheetah/qmix_cheetah
python -m scripts.baselines.train_maxVQ_simba --config-name baselines_configs/maxVQ/cheetah/maxVQ_cheetah
```

**Oracle paths for baselines:** Each baseline’s run config (e.g. `baselines_configs/loki/cheetah/loki_cheetah.yaml`) has `defaults:` that include the **same** task oracle config as CurrimaxAdv for that task (e.g. `- /oracles/dmc/cheetah_run`). So `oracles_dict` is shared: all methods use the same set of oracle checkpoints for fair comparison. Hyperparameters for each baseline are in the corresponding base config under `config/base_configs/` (e.g. `loki_dmc.yaml`, `cup_dmc.yaml`). Override them in the baseline’s run config or via the command line.

