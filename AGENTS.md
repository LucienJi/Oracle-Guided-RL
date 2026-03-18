# Repository Guidelines

## Core Principles
- Fail-Fast over silent failure
- No hidden assumptions
- Reproducibility first
- Validation is mandatory
- Worklog is the source of truth

---

## Working Style
- Before major edits, write/update `docs/codex_plan.md`
- After each completed milestone, append to `docs/codex_worklog.md`
- Plans may evolve; worklog must remain factual
- Do not claim something is done unless validated
- Prefer small, verifiable iterations

---

## Plan Management
- `docs/codex_plan.md` reflects the CURRENT plan only
- Overwrite it when the plan changes significantly
- Keep it concise (≤ 50 lines)
- Focus on actionable steps, not discussion

---

## Worklog Requirements
Each entry in `docs/codex_worklog.md` must include:

- Date/time
- Objective
- Key findings (**required, not optional**)
- Files changed
- Commands run
- Validation result
- Outcome (what actually happened, not what was expected)
- Next step
- Blockers

---

## Validation Rules (STRICT)

- Every non-trivial code change MUST be validated
- Validation should be:
  - unit test (`/tests`)
  - script run
  - or reproducible command

- If no validation:
  → explicitly write `"Not validated"`

- If partial validation:
  → specify EXACTLY what was checked

- NEVER claim success without evidence

---

## Testing Policy

- All new logic should have corresponding tests under `/tests`
- Prefer:
  - minimal reproducible test scripts
  - fast execution (fail-fast)

- Test should cover:
  - expected behavior
  - edge cases (if relevant)

---

## Fail-Fast Policy

- Prefer explicit failure over silent handling
- Avoid `try/except` unless:
  - error is expected AND
  - recovery is explicitly defined

- Do NOT suppress errors
- Do NOT add fallback logic that hides bugs

---

## Configuration Policy

- Do NOT use `argparse`
- Use `hydra` + `yaml` for all configurations

- Configs must be:
  - version-controllable
  - human-readable
  - reproducible

- Avoid hard-coded parameters in code

---

## Environment & Reproducibility

- The project MUST provide a runnable environment
- we can use "conda activate oracle"

- The agent should:
  - update environment config when dependencies change
  - ensure scripts run in a clean environment


---

## Anti-Drift Rule

- If implementation deviates from the plan:
  → MUST explicitly record the deviation in the worklog

Include:
- what changed
- why it changed

---

## Execution Philosophy

- Prefer correctness over completeness
- Prefer simple solutions over complex ones
- Prefer explicit over implicit
- Stop early if assumptions are unclear → log blocker


## Project Structure & Module Organization
Core Python code lives in `algo/`, `env/`, `model/`, `data_buffer/`, and `viz/`. Training entry points are in `scripts/` and `scripts/baselines/`. Hydra configs are grouped by domain in `config/` (`dmc/`, `box2d/`, `metaworld/`, `base_configs/`, `oracles/`). Third-party editable dependencies live under `third_party/`. Generated artifacts such as `checkpoints/`, `outputs/`, `eval/`, videos, and `config/paths_local.yaml` are ignored and should stay out of commits.

## Build, Test, and Development Commands
- `conda env create -f requirements/oracles.yaml`: create the Python 3.10 environment.
- `conda activate oracles && bash setup_paths.sh`: create the local path override used by configs.
- `bash install.sh`: install the Conda env plus editable third-party packages.
- `python scripts/train_bc.py`: run a default Hydra training entry point.
- `python scripts/train_CurrimaxAdv.py --config-name dmc/CurrimaxAdv_cheetah`: run a specific training config.
- `bash launch_train.sh train_bc 1 --config-name dmc/bc_humanoid`: submit a SLURM job segment.
- `python test.py` or `python test_mw.py`: run environment smoke tests and rendering checks.

## Coding Style & Naming Conventions
Use Python 3.10, 4-space indentation, and snake_case for modules, functions, config keys, and experiment names. Follow the existing import style: standard library, third-party, then local modules. Keep training scripts small and config-driven; put reusable logic in `algo/`, `env/`, or `model/` instead of duplicating it in `scripts/`. Config files follow `<method>_<task>.yaml` patterns such as `bc_hopper.yaml` or `CurrimaxAdv_assembly.yaml`.

## Testing Guidelines
There is no formal unit-test tree yet; current coverage is smoke-test based. Add quick checks as `test_*.py` scripts that can run from the repo root without manual path edits. When changing environments or rendering, include a command to reproduce the check and note any GPU or `MUJOCO_GL=egl` requirement.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects (`new configs`, `prepare for the work on other cluster`). Keep commit titles brief and specific to one change. For pull requests, include: purpose, affected configs/scripts, required data or checkpoints, exact run/test commands, and sample metrics or screenshots when behavior changes.

## Configuration & Environment Tips
Do not hardcode cluster-specific paths. Use `config/paths.yaml`, `config/paths_local.yaml`, or `ORACLES_PROJECT_ROOT`. Keep large artifacts and downloaded third-party code out of Git.
