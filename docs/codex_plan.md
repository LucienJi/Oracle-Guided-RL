## Current Plan

Date: 2026-03-18
Objective: Prepare a professor-facing, submission-ready DMC-only package with a local smoke path, a documented Slurm path, and honest setup/validation documentation.

Status: Completed

Completed steps:
1. Cleaned the DMC execution path so `scripts/train_simba.py` and `scripts/train_CurrimaxAdv.py` no longer hard-import optional benchmark packages through `env/env_utils.py`.
2. Added the review-facing demo path: `scripts/smoke_dmc.py`, submission configs, `setup_paths.sh`, examples, and a representative Slurm script.
3. Rewrote `README.md`, updated `INSTALL.md`, added `report_draft.md`, and added `SUBMISSION_CHECKLIST.md`.
4. Added proportional tests for the submission configs and smoke/import path.
5. Ran static validation and recorded the environment-backed validation limits in `docs/codex_worklog.md`.
