# Codex Worklog

## 2026-03-18 - Batch 1 - Safe cleanup

Changes:
- Deleted `debug.ipynb`
- Deleted `data_buffer/replay_buffer_old.py`
- Deleted `algo/oracles/CurrimaxAdv_backup.py`
- Deleted `algo/oracles/CurrimaxAdv_test_backup.py`
- Deleted deprecated tracked algorithm files under `algo/deprecated/`
- Deleted deprecated tracked runner files under `scripts/deprecated/`

Validation:
- Ran `python -m compileall algo env model data_buffer scripts muon_pkg viz`
- Ran `git grep -n -E 'replay_buffer_old|CurrimaxAdv_backup|CurrimaxAdv_test_backup|train_CurrimaxAdv_re|train_CurrimaxAdv_re_v2|train_SharedmaxAdv|train_maxAdv' -- . || true`
- Result: static compilation passed and no tracked references remained

Environment notes:
- `conda` is not available in the current shell, so environment-backed runtime validation could not be run in this batch

Files changed:
- Deleted `debug.ipynb`
- Deleted `data_buffer/replay_buffer_old.py`
- Deleted `algo/oracles/CurrimaxAdv_backup.py`
- Deleted `algo/oracles/CurrimaxAdv_test_backup.py`
- Deleted `algo/deprecated/CurrimaxAdv_re.py`
- Deleted `algo/deprecated/CurrimaxAdv_re_v2.py`
- Deleted `algo/deprecated/SharedmaxAdv.py`
- Deleted `algo/deprecated/maxAdv.py`
- Deleted `scripts/deprecated/train_CurrimaxAdv_re.py`
- Deleted `scripts/deprecated/train_CurrimaxAdv_re_v2.py`
- Deleted `scripts/deprecated/train_SharedmaxAdv.py`
- Deleted `scripts/deprecated/train_maxAdv.py`
- Added `docs/codex_worklog.md`

## 2026-03-18 - Batch 2 - Setup hardening

Changes:
- Added canonical root environment file `environment.yml`
- Declared tracked visualization dependencies in `requirements/base.txt`
- Updated `install.sh` to use `environment.yml`, fail fast on missing `third_party/*`, and stop editing `~/.bashrc`

Validation:
- Ran `bash -n install.sh`
- Ran `python -m compileall algo env model data_buffer scripts muon_pkg viz`
- Checked that `requirements/base.txt` contains `seaborn` and `rliable`
- Checked that `install.sh` references `environment.yml` and the required `third_party/*` directories

Reproducibility blocker:
- This repository requires local clones under `third_party/`, but those repos are not versioned by the main repository and are not git submodules.
- Observed local origins / commits in this checkout:
  - `third_party/CARL`: `https://github.com/automl/CARL.git` @ `2adf54cceff3b794c0ee330debf44bc428c2015d`
  - `third_party/Metaworld`: `https://github.com/Farama-Foundation/Metaworld.git` @ `066c391a20a48f34b38c1c2faef85aa4f5068b0d`
  - `third_party/HighwayEnv`: `https://github.com/Farama-Foundation/HighwayEnv.git` @ `75342a1b77e7ed33b99330a356890ffe31fbf9cb`
  - `third_party/myosuite`: `https://github.com/facebookresearch/myosuite.git` @ `300058ea09f34f8598e4f4a6e3765c34676f14c4`
- Minimal fix proposal: track these dependencies via git submodules or add a pinned bootstrap script that clones exact commits.
- Current cleanup pass keeps structure unchanged and makes the installer fail fast instead of silently assuming those directories exist.

Environment notes:
- `conda` is not available in the current shell, so `conda env create -f environment.yml --dry-run` could not be run here

Files changed:
- Added `environment.yml`
- Updated `requirements/base.txt`
- Updated `install.sh`

## 2026-03-18 - Batch 3 - Entry-point portability and minimal tests

Changes:
- Fixed direct-script repo-root resolution in tracked top-level entry points under `scripts/`
- Added tracked smoke tests under `tests/`
- Preserved `CurrimaxAdv_test.py` and its runner scripts

Validation:
- Ran `python -m compileall algo env model data_buffer scripts muon_pkg viz tests`
- Attempted `pytest tests -q`
- Result: `compileall` passed; `pytest` could not run in the current shell because `pytest` is not installed here (`/usr/bin/bash: pytest: command not found`)

Files changed:
- Updated `scripts/train_CurrimaxAdv.py`
- Updated `scripts/train_CurrimaxAdv_simba.py`
- Updated `scripts/train_CurrimaxAdv_test.py`
- Updated `scripts/train_CurrimaxAdv_test_simba.py`
- Updated `scripts/train_bc.py`
- Updated `scripts/train_simba.py`
- Updated `scripts/test.py`
- Added `tests/test_config_smoke.py`
- Added `tests/test_model_smoke.py`

## 2026-03-18 - Batch 4 - Review-facing docs and path cleanup

Changes:
- Rewrote `README.md` as the primary review-facing guide
- Rewrote `INSTALL.md` to align with the canonical root environment path and document the unversioned `third_party/` blocker
- Replaced machine-specific tracked example paths in `config/paths_local.yaml.example`
- Replaced machine-specific visualization paths in `viz/viz_config/default.yaml` and `viz/viz_config/plot_saved.yaml`

Validation:
- Ran `python -m compileall algo env model data_buffer scripts muon_pkg viz tests`
- Attempted `python -m pytest tests -q`
- Checked that `README.md` does not reference ignored local launchers or `quick_setup_expanse.sh`
- Checked that the rewritten tracked docs/configs no longer contain the previous `/share/data/...` or `/expanse/.../Oracle-Guided-RL` hardcoded paths
- Result: `compileall` passed; `python -m pytest` could not run in the current shell because `pytest` is not installed (`/usr/bin/python: No module named pytest`)

Files changed:
- Updated `README.md`
- Updated `INSTALL.md`
- Updated `config/paths_local.yaml.example`
- Updated `viz/viz_config/default.yaml`
- Updated `viz/viz_config/plot_saved.yaml`

## 2026-03-18 - Batch 5 - Low-risk code polish

Changes:
- Removed low-risk unused imports from tracked core modules
- Made `scripts/test.py` non-interactive by removing `breakpoint()` and wrapping it in `main()`
- Removed broad exception masking from `scripts/update_config_paths.py` so it now fails fast on file errors

Validation:
- Ran `python -m compileall algo env model data_buffer scripts muon_pkg viz tests`
- Checked that tracked `scripts/test.py` no longer contains `breakpoint()`
- Attempted `python -m pytest tests -q`
- Result: `compileall` passed; no `breakpoint()` remained in tracked `scripts/test.py`; `python -m pytest` still could not run because `pytest` is not installed in the current shell (`/usr/bin/python: No module named pytest`)

Files changed:
- Updated `algo/algo_utils.py`
- Updated `algo/base_algo.py`
- Updated `algo/baselines/cup.py`
- Updated `algo/baselines/loki.py`
- Updated `algo/baselines/maps.py`
- Updated `algo/baselines/maxVQ.py`
- Updated `algo/baselines/qmix.py`
- Updated `algo/oracles/CurrimaxAdv.py`
- Updated `algo/oracles/CurrimaxAdv_test.py`
- Updated `algo/oracles/NoAdv.py`
- Updated `algo/oracles/oracle_loading.py`
- Updated `model/mlp.py`
- Updated `model/oracle_wrappers.py`
- Updated `model/simba.py`
- Updated `model/simba_base.py`
- Updated `model/simba_share.py`
- Updated `scripts/update_config_paths.py`
- Updated `scripts/test.py`

## 2026-03-18 18:50:02 CDT - Batch 6 - Translate Chinese comments to English

Objective:
- Translate Chinese comments/docstrings in tracked repo files into English without changing runtime behavior.

Key findings:
- Chinese text was confined to a small set of tracked Python, YAML, and shell files.
- `config/base_configs/bc_box2d.yaml` contained a malformed stray first line unrelated to the translation request; it was removed to restore a valid YAML header.
- After the edit pass, a tracked-file scan found no remaining Han characters anywhere in tracked text files.

Files changed:
- Added `docs/codex_plan.md`
- Updated `algo/algo_utils.py`
- Updated `config/base_configs/bc_box2d.yaml`
- Updated `config/paths_local.yaml.example`
- Updated `env/env_utils.py`
- Updated `install.sh`
- Updated `model/simba_base.py`
- Updated `third_party/customized_box2d/HybridBipedalWalker.py`
- Updated `third_party/customized_box2d/HybridCar.py`
- Updated `third_party/customized_box2d/HybridLunarLand.py`
- Updated `third_party/customized_box2d/HybridWeather.py`
- Updated `docs/codex_worklog.md`

Commands run:
- `python - <<'PY' ... git ls-files tracked-file Han scan ... PY`
- `sed -n '1,220p' config/base_configs/bc_box2d.yaml`
- `sed -n '220,245p' algo/algo_utils.py`
- `sed -n '485,645p' env/env_utils.py`
- `sed -n '1,110p' install.sh`
- `sed -n '1,80p' model/simba_base.py`
- `sed -n '1,20p' third_party/customized_box2d/HybridBipedalWalker.py`
- `sed -n '275,760p' third_party/customized_box2d/HybridBipedalWalker.py`
- `sed -n '48,95p' third_party/customized_box2d/HybridCar.py`
- `sed -n '40,520p' third_party/customized_box2d/HybridCar.py`
- `sed -n '220,360p' third_party/customized_box2d/HybridLunarLand.py`
- `sed -n '140,520p' third_party/customized_box2d/HybridWeather.py`
- `python -m compileall algo/algo_utils.py env/env_utils.py model/simba_base.py third_party/customized_box2d/HybridBipedalWalker.py third_party/customized_box2d/HybridCar.py third_party/customized_box2d/HybridLunarLand.py third_party/customized_box2d/HybridWeather.py`
- `bash -n install.sh`
- `sed -n '1,5p' config/base_configs/bc_box2d.yaml`

Validation result:
- Repo-wide tracked-file Han scan returned `NO_HAN_TEXT_FOUND`
- `python -m compileall ...` passed for all changed Python files
- `bash -n install.sh` passed

Outcome:
- Translated all discovered Chinese comments/docstrings to English across tracked files.
- Translated the remaining Chinese install-script messages to English so tracked files are free of Han text.
- Deviation from the original plan: removed a malformed stray header line from `config/base_configs/bc_box2d.yaml` because the scan exposed invalid file content and leaving it in place would keep the config broken.

Next step:
- No further repository changes are required for this task unless a narrower translation style pass is requested.

Blockers:
- None

## 2026-03-18 19:21:46 CDT - Batch 7 - DMC submission path hardening

Objective:
- Prepare the DMC-only review path so the main entry points are importable without optional benchmark dependencies and add a concrete local smoke/demo path.

Key findings:
- `env/env_utils.py` eagerly imported optional benchmark packages and Box2D modules at import time, which would break the DMC-only submission path before any CLI help or config loading could happen.
- The tracked docs referenced `setup_paths.sh`, but the script was missing from the repository.
- The current shell still does not provide `pytest`, so runtime validation remains limited to static checks here.

Files changed:
- Updated `docs/codex_plan.md`
- Updated `env/env_utils.py`
- Updated `scripts/train_simba.py`
- Updated `scripts/train_CurrimaxAdv.py`
- Added `scripts/smoke_dmc.py`
- Added `config/dmc/submission_smoke_cartpole.yaml`
- Added `config/dmc/submission_currimaxadv_smoke.yaml`
- Added `setup_paths.sh`
- Added `tests/test_submission_package.py`
- Updated `docs/codex_worklog.md`

Commands run:
- `sed -n '1,80p' env/env_utils.py`
- `sed -n '180,320p' env/env_utils.py`
- `sed -n '430,560p' env/env_utils.py`
- `sed -n '1,140p' scripts/train_simba.py`
- `sed -n '1,220p' scripts/train_CurrimaxAdv.py`
- `python -m compileall env/env_utils.py scripts/train_simba.py scripts/train_CurrimaxAdv.py scripts/smoke_dmc.py tests/test_submission_package.py`
- `bash -n setup_paths.sh`
- `python -m pytest tests/test_submission_package.py -q`

Validation result:
- `python -m compileall ...` passed for the updated DMC entrypoints, smoke script, and test file
- `bash -n setup_paths.sh` passed
- `python -m pytest tests/test_submission_package.py -q` could not run because `pytest` is not installed in the current shell (`/usr/bin/python: No module named pytest`)

Outcome:
- The DMC review path now defers optional dependency imports until the corresponding benchmark factories are used.
- Added a review-facing smoke script, two submission configs, and the missing tracked path-setup helper.
- Added proportional tests for the submission package shape, pending a shell with `pytest` installed.

Next step:
- Rewrite the professor-facing README/report/checklist and add the representative Slurm/example documentation.

Blockers:
- `pytest` is unavailable in the current shell

## 2026-03-18 19:25:42 CDT - Batch 8 - Professor-facing docs and validation

Objective:
- Finish the DMC-only submission package with professor-facing documentation, a representative Slurm script, and a final validation pass.

Key findings:
- `config/paths.yaml` documented a `paths_local.yaml` override path but did not actually consult `paths_local.project_root`; the config was updated so the tracked `setup_paths.sh` helper now has a real effect.
- CLI help for the new entry points could not be exercised in the current shell because `hydra` is not installed here.
- Static validation succeeded for the edited Python and shell entry points; runtime-heavy validation still requires the project Conda environment.

Files changed:
- Updated `config/paths.yaml`
- Updated `README.md`
- Updated `INSTALL.md`
- Added `examples/README.md`
- Added `slurm/dmc_train_simba.sbatch`
- Added `report_draft.md`
- Added `SUBMISSION_CHECKLIST.md`
- Updated `docs/codex_worklog.md`

Commands run:
- `python -m compileall env/env_utils.py scripts/train_simba.py scripts/train_CurrimaxAdv.py scripts/smoke_dmc.py tests/test_submission_package.py`
- `bash -n setup_paths.sh`
- `bash -n slurm/dmc_train_simba.sbatch`
- `python -m scripts.train_simba --help`
- `python -m scripts.train_CurrimaxAdv --help`
- `python -m scripts.smoke_dmc --help`
- `python -m pytest tests/test_submission_package.py -q`
- `bash setup_paths.sh`
- `sed -n '1,20p' config/paths_local.yaml`

Validation result:
- `python -m compileall ...` passed
- `bash -n setup_paths.sh` passed
- `bash -n slurm/dmc_train_simba.sbatch` passed
- `bash setup_paths.sh` succeeded and wrote `config/paths_local.yaml`
- `python -m scripts.train_simba --help`, `python -m scripts.train_CurrimaxAdv --help`, and `python -m scripts.smoke_dmc --help` failed because `hydra` is not installed in the current shell
- `python -m pytest tests/test_submission_package.py -q` failed because `pytest` is not installed in the current shell

Outcome:
- Added a complete professor-facing DMC submission package: narrowed README, installation notes, example commands, report draft, Slurm script, and checklist.
- The path setup helper is now functionally connected to `config/paths.yaml`.
- The package is ready for review from a repository-organization perspective, with explicit notes about what was and was not verified.

Next step:
- Activate the `oracles` Conda environment and rerun the CLI help, smoke test, and pytest commands if environment-backed verification is required before handing off the package.

Blockers:
- The current shell is missing `hydra` and `pytest`
