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
