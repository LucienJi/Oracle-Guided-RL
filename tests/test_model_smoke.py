from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.simba import DeterministicSimbaPolicy


def test_deterministic_simba_policy_forward():
    model_args = SimpleNamespace(
        num_blocks=1,
        hidden_dim=32,
        scaler_init=0.3,
        scaler_scale=1.0,
        alpha_init=0.3,
        alpha_scale=1.0,
        c_shift=4.0,
        noise_clip=0.5,
        action_high=[1.0, 1.0],
        action_low=[-1.0, -1.0],
    )
    policy = DeterministicSimbaPolicy(obs_shape=(5,), action_dim=2, model_args=model_args)
    obs = torch.zeros(3, 5, dtype=torch.float32)

    actions = policy.get_eval_action(obs)

    assert actions.shape == (3, 2)
    assert torch.all(actions <= 1.0 + 1e-6)
    assert torch.all(actions >= -1.0 - 1e-6)
