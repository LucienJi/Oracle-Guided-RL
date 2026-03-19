from pathlib import Path
import importlib
import sys

import pytest

hydra = pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _compose(config_name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "config")):
        return compose(config_name=config_name)


def test_compose_submission_simba_smoke():
    cfg = _compose("dmc/submission_smoke_cartpole")
    assert cfg.env.domain_name == "cartpole"
    assert cfg.env.task_name == "swingup_sparse"
    assert cfg.smoke.workflow == "simba"
    assert cfg.training.use_wandb is False
    assert cfg.training.use_compile is False


def test_compose_submission_currimaxadv_smoke():
    cfg = _compose("dmc/submission_currimaxadv_smoke")
    assert cfg.env.domain_name == "cartpole"
    assert cfg.training.n_oracles == len(cfg.oracles_dict)
    assert cfg.smoke.workflow == "currimaxadv"
    assert all(value is None for value in cfg.oracles_dict.values())


def test_submission_smoke_script_imports():
    module = importlib.import_module("scripts.smoke_dmc")
    assert callable(module.run_smoke)


def test_env_utils_import_is_available_for_dmc_submission():
    module = importlib.import_module("env.env_utils")
    assert hasattr(module, "make_env_and_eval_env_from_cfg")
