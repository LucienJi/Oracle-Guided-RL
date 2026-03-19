# Submission Checklist

## Required deliverables

- [x] Program code is included
- [x] Professor-facing project overview is included in `README.md`
- [x] Explanation of how the program works is included in `README.md` and `report_draft.md`
- [x] Python compilation/setup instructions are included
- [x] Run/use instructions are included for local and Slurm usage
- [x] Input/output format description is included
- [x] Sample commands are included
- [x] A short report draft is included
- [x] A representative Slurm script is included
- [x] Lightweight smoke tests/configs are included

## Code/package cleanup

- [x] DMC-only submission scope chosen to avoid over-packaging the full research workspace
- [x] Main entry points narrowed to `scripts/train_simba.py` and `scripts/train_CurrimaxAdv.py`
- [x] Optional benchmark imports deferred so they no longer break the DMC path at import time
- [x] Missing tracked helper `setup_paths.sh` added
- [x] Submission smoke configs added under `config/dmc/`
- [x] Local smoke helper added as `scripts/smoke_dmc.py`
- [x] Targeted argument/config validation added to the main DMC entry points

## Validation status

Verified in the current shell:

- [x] `python -m compileall env/env_utils.py scripts/train_simba.py scripts/train_CurrimaxAdv.py scripts/smoke_dmc.py tests/test_submission_package.py`
- [x] `bash -n setup_paths.sh`
- [x] `bash setup_paths.sh`
- [x] `bash -n slurm/dmc_train_simba.sbatch`

Not verified in the current shell:

- [ ] `python -m pytest tests -q`
- [ ] `python -m scripts.train_simba --help`
- [ ] `python -m scripts.train_CurrimaxAdv --help`
- [ ] `python -m scripts.smoke_dmc --config-name dmc/submission_smoke_cartpole`
- [ ] Full end-to-end training run

Reason not fully verified:

- The current shell does not provide the project runtime environment (`hydra` and `pytest` were unavailable when command validation was attempted), so only static validation and shell-script checks could be completed here.

## Honest scope notes

- [x] Cluster/Slurm usage is documented explicitly
- [x] Non-DMC benchmark families are documented as out of scope for the submission package
- [x] The package does not claim capabilities that were not validated
