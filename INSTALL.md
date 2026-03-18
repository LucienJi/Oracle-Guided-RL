# Installation Notes

This file supplements `README.md` with the benchmark dependency details that matter for reproducibility.

## Canonical environment path

```bash
conda env create -f environment.yml
conda activate oracles
bash setup_paths.sh
```

Use `environment.yml` as the canonical environment definition. `requirements/oracles.yaml` is kept for compatibility with existing workflows, but the review-facing path should start from the repo root.

## Optional benchmark dependencies

DeepMind Control runs can start after the Conda environment is created.

The following local clones are required only for the corresponding benchmark families:

```bash
pip install -e third_party/CARL
pip install -e third_party/Metaworld
pip install -e third_party/HighwayEnv
pip install -e third_party/myosuite
```

If one of those directories is missing, `install.sh` now fails immediately instead of silently continuing.

## Reproducibility blocker

The main repository does not version these `third_party/` dependencies and does not register them as git submodules. In the current checkout, the observed local references are:

- `third_party/CARL`: `https://github.com/automl/CARL.git` @ `2adf54cceff3b794c0ee330debf44bc428c2015d`
- `third_party/Metaworld`: `https://github.com/Farama-Foundation/Metaworld.git` @ `066c391a20a48f34b38c1c2faef85aa4f5068b0d`
- `third_party/HighwayEnv`: `https://github.com/Farama-Foundation/HighwayEnv.git` @ `75342a1b77e7ed33b99330a356890ffe31fbf9cb`
- `third_party/myosuite`: `https://github.com/facebookresearch/myosuite.git` @ `300058ea09f34f8598e4f4a6e3765c34676f14c4`

Minimal fix: replace this implicit requirement with pinned git submodules or a tracked bootstrap script that clones exact commits.

## Validation commands

```bash
python -m compileall algo env model data_buffer scripts muon_pkg viz tests
python -m pytest tests -q
```

## Notes

- `setup_paths.sh` creates `config/paths_local.yaml`; do not hardcode machine paths in tracked configs.
- `install.sh` no longer edits `~/.bashrc`. Set `MUJOCO_GL=egl` explicitly in the shell or job script when needed.
- Review-facing runs should use tracked Python entry points, not ignored local launcher shells.
