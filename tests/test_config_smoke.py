from pathlib import Path
import sys

import pytest

hydra = pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _compose(config_name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "config")):
        return compose(config_name=config_name)


def test_compose_simba_cartpole_sparse():
    cfg = _compose("dmc/simba_cartpole_sparse")
    assert cfg.env.domain_name == "cartpole"
    assert cfg.env.task_name == "swingup_sparse"
    assert cfg.training.total_timesteps > 0
    assert cfg.training.batch_size > 0


def test_compose_currimaxadv_box2d():
    cfg = _compose("box2d/CurrimaxAdv_bipedal_learner")
    assert cfg.training.n_oracles == len(cfg.oracles_dict)
    assert cfg.training.total_timesteps > 0
    assert cfg.rl_agent_args.hidden_dim > 0
