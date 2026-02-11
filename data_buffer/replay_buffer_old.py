import datetime
import io
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def save_episode(episode: Dict[str, np.ndarray], fn) -> None:
    """
    Save an episode dict to disk as a compressed npz.
    """
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn) -> Dict[str, np.ndarray]:
    with fn.open("rb") as f:
        episode = np.load(f)
        return {k: episode[k] for k in episode.keys()}


class ReplayBuffer:
    """
    A simple memory-based replay buffer with n-step sampling.

    Key simplification vs previous version:
    - `observations` and `next_observations` are single arrays/tensors (NOT dicts).

    Public API used by the repo:
    - add_transition(transition)
    - sample(batch_size)
    - size() / __len__()
    - action_mask_sample(batch_size, target_action_mask, on_policy_ratio=1.0)
    - recency_sample(batch_size, current_step, recency_lambda)
    """

    def __init__(
        self,
        replay_dir: str,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        capacity: int = 100000,
        nstep: int = 1,
        discount: float = 0.99,
        device: Optional[str] = None,
        stream_to_local_buffer: bool = True,
        save_episodes_to_disk: bool = True,
    ):
        import pathlib

        self.replay_dir = pathlib.Path(replay_dir)
        self.replay_dir.mkdir(exist_ok=True, parents=True)

        self.observation_shape = tuple(observation_shape)
        self.action_dim = int(action_dim)

        self._max_size = int(capacity)
        self.nstep = int(nstep)
        self.discount = float(discount)
        self.stream_to_local_buffer = bool(stream_to_local_buffer)
        self.save_episodes_to_disk = bool(save_episodes_to_disk)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Episode (python) staging area
        self._current_episode = defaultdict(list)
        self._episode_step_count = 0

        # Local replay buffer (tensors on device)
        self._local_replay_buffer: Optional[Dict[str, torch.Tensor]] = None
        self._buffer_size = 0
        self._total_transitions = 0

        # Episode boundary tracking: list[(start_idx, end_idx)] in buffer coordinates
        self._episode_boundaries = []
        self._valid_episodes = []

        # Global cache of valid n-step start indices (on device)
        self._valid_starts = torch.empty((self._max_size,), device=self.device, dtype=torch.long)
        self._valid_starts_count = 0

        # Optional per-action_mask cache (rebuilt lazily)
        self._action_mask_valid_starts: Dict[int, torch.Tensor] = {}
        self._action_mask_valid_starts_count: Dict[int, int] = {}
        self._action_mask_cache_version: Dict[int, int] = {}

        print(f"ReplayBuffer (device={self.device}) initialized with capacity: {self._max_size}")

        # Load existing episodes
        self._preload()

    # -----------------------
    # Size helpers
    # -----------------------
    def size(self) -> int:
        return int(self._buffer_size)

    def __len__(self) -> int:
        return int(self._buffer_size)

    # -----------------------
    # Internal helpers
    # -----------------------
    def _reset_action_mask_cache(self) -> None:
        self._action_mask_valid_starts = {}
        self._action_mask_valid_starts_count = {}
        self._action_mask_cache_version = {}

    def _ensure_action_mask_cache(self, mask_val: int) -> None:
        mask_val = int(mask_val)
        if mask_val not in self._action_mask_valid_starts:
            self._action_mask_valid_starts[mask_val] = torch.empty((0,), device=self.device, dtype=torch.long)
            self._action_mask_valid_starts_count[mask_val] = 0
            self._action_mask_cache_version[mask_val] = -1

    def _cache_add_valid_start(self, start_idx: int) -> None:
        start_idx = int(start_idx)
        c = int(self._valid_starts_count)
        if c >= self._max_size:
            return
        self._valid_starts[c] = start_idx
        self._valid_starts_count = c + 1

    def _cache_add_valid_starts_batch(self, start_indices: torch.Tensor) -> None:
        if start_indices.numel() == 0:
            return
        start_indices = start_indices.to(torch.long)
        n = int(start_indices.numel())
        c = int(self._valid_starts_count)
        if c >= self._max_size:
            return
        end = min(self._max_size, c + n)
        take = end - c
        if take <= 0:
            return
        self._valid_starts[c:end] = start_indices[:take]
        self._valid_starts_count = end

    def _initialize_local_buffer(self) -> None:
        self._local_replay_buffer = {
            "observations": torch.empty((self._max_size, *self.observation_shape), dtype=torch.float32, device=self.device),
            "next_observations": torch.empty((self._max_size, *self.observation_shape), dtype=torch.float32, device=self.device),
            "actions": torch.empty((self._max_size, self.action_dim), dtype=torch.float32, device=self.device),
            "rewards": torch.empty((self._max_size,), dtype=torch.float32, device=self.device),
            "dones": torch.empty((self._max_size,), dtype=torch.bool, device=self.device),
            # Always present for consistency
            "action_mask": torch.empty((self._max_size,), dtype=torch.int32, device=self.device),
            "clean_action": torch.empty((self._max_size, self.action_dim), dtype=torch.float32, device=self.device),
            "global_step": torch.empty((self._max_size,), dtype=torch.int32, device=self.device),
        }

    def _add_step_to_buffer(self, transition: Dict[str, Any]) -> None:
        if self._local_replay_buffer is None:
            self._initialize_local_buffer()

        if self._buffer_size >= self._max_size:
            print(f"Replay buffer at capacity ({self._max_size}). Skipping step add.")
            return

        idx = self._buffer_size

        obs = np.asarray(transition["observations"], dtype=np.float32).reshape(self.observation_shape)
        next_obs = np.asarray(transition["next_observations"], dtype=np.float32).reshape(self.observation_shape)

        self._local_replay_buffer["observations"][idx] = torch.from_numpy(obs).to(self.device, dtype=torch.float32, non_blocking=True)
        self._local_replay_buffer["next_observations"][idx] = torch.from_numpy(next_obs).to(self.device, dtype=torch.float32, non_blocking=True)

        act = np.asarray(transition["actions"], dtype=np.float32).reshape(self.action_dim)
        self._local_replay_buffer["actions"][idx] = torch.from_numpy(act).to(self.device, dtype=torch.float32, non_blocking=True)

        rew = transition["rewards"]
        rew_t = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
        self._local_replay_buffer["rewards"][idx] = rew_t

        done = transition["dones"]
        done_t = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        self._local_replay_buffer["dones"][idx] = done_t

        # Optional fields
        mask = transition.get("action_mask", 1)
        self._local_replay_buffer["action_mask"][idx] = torch.as_tensor(mask, device=self.device, dtype=torch.int32)

        clean_action = transition.get("clean_action", act)
        ca = np.asarray(clean_action, dtype=np.float32).reshape(self.action_dim)
        self._local_replay_buffer["clean_action"][idx] = torch.from_numpy(ca).to(self.device, dtype=torch.float32, non_blocking=True)

        gs = transition.get("global_step", 0)
        self._local_replay_buffer["global_step"][idx] = torch.as_tensor(gs, device=self.device, dtype=torch.int32)

        # Update sizes
        self._buffer_size += 1
        self._total_transitions += 1

        # Update episode boundaries
        if self._episode_step_count == 0:
            start_idx = idx
            end_idx = idx + 1
            self._episode_boundaries.append((start_idx, end_idx))
            episode_idx = len(self._episode_boundaries) - 1
            if (end_idx - start_idx) >= self.nstep and episode_idx not in self._valid_episodes:
                self._valid_episodes.append(episode_idx)
        else:
            start_idx, end_idx = self._episode_boundaries[-1]
            end_idx = end_idx + 1
            self._episode_boundaries[-1] = (start_idx, end_idx)
            episode_idx = len(self._episode_boundaries) - 1
            if (end_idx - start_idx) >= self.nstep and episode_idx not in self._valid_episodes:
                self._valid_episodes.append(episode_idx)

        # Each new appended step creates at most one new valid n-step window start.
        start_idx, end_idx = self._episode_boundaries[-1]
        ep_len = end_idx - start_idx
        if ep_len >= self.nstep:
            new_start = end_idx - self.nstep
            if new_start >= start_idx:
                self._cache_add_valid_start(new_start)

    # -----------------------
    # Public write path
    # -----------------------
    def add_transition(self, transition: Dict[str, Any]) -> None:
        """
        transition must contain:
          - observations: np.ndarray (observation_shape)
          - next_observations: np.ndarray (observation_shape)
          - actions: np.ndarray (action_dim)
          - rewards: float
          - dones: bool
        optional:
          - truncated: bool
          - action_mask, clean_action, global_step
        """
        self._current_episode["observations"].append(np.asarray(transition["observations"], dtype=np.float32).reshape(self.observation_shape))
        self._current_episode["next_observations"].append(np.asarray(transition["next_observations"], dtype=np.float32).reshape(self.observation_shape))
        self._current_episode["actions"].append(np.asarray(transition["actions"], dtype=np.float32).reshape(self.action_dim))
        self._current_episode["rewards"].append(float(transition["rewards"]))
        self._current_episode["dones"].append(bool(transition["dones"]))

        if "action_mask" in transition:
            self._current_episode["action_mask"].append(int(transition["action_mask"]))
        if "clean_action" in transition:
            self._current_episode["clean_action"].append(np.asarray(transition["clean_action"], dtype=np.float32).reshape(self.action_dim))
        if "global_step" in transition:
            self._current_episode["global_step"].append(int(transition["global_step"]))

        if self.stream_to_local_buffer:
            self._add_step_to_buffer(transition)

        self._episode_step_count += 1

        if transition["dones"] or bool(transition.get("truncated", False)):
            self._store_current_episode(add_to_local_buffer=not self.stream_to_local_buffer)

    # -----------------------
    # Disk preload / episode finalization
    # -----------------------
    def _preload(self) -> None:
        self._buffer_size = 0
        self._total_transitions = 0
        self._episode_boundaries = []
        self._valid_episodes = []
        self._valid_starts_count = 0
        self._reset_action_mask_cache()

        episode_files = sorted(list(self.replay_dir.glob("*.npz")))
        if len(episode_files) == 0:
            return

        print(f"Found {len(episode_files)} episode files to preload...")
        for i, fn in enumerate(episode_files):
            try:
                episode = load_episode(fn)
                eps_len = int(len(episode["actions"]))
                if self._local_replay_buffer is None:
                    self._initialize_local_buffer()
                self._add_episode_to_buffer(episode, eps_len)
                self._total_transitions += eps_len

                if (i + 1) % 100 == 0 or (i + 1) == len(episode_files):
                    print(
                        f"Preloaded {i + 1}/{len(episode_files)} episodes, "
                        f"{self._total_transitions} total transitions, "
                        f"buffer size: {self._buffer_size}/{self._max_size}"
                    )

                if self._buffer_size >= self._max_size:
                    print(f"Buffer full ({self._max_size} transitions), stopping preload.")
                    break
            except Exception as e:
                print(f"Error loading episode {fn}: {e}")
                continue

        # Episodes valid for n-step sampling (by boundaries)
        self._valid_episodes = []
        for ep_idx, (s, e) in enumerate(self._episode_boundaries):
            if (e - s) >= self.nstep:
                self._valid_episodes.append(ep_idx)

        print(f"Episodes valid for {self.nstep}-step sampling: {len(self._valid_episodes)}/{len(self._episode_boundaries)}")

    def _add_episode_to_buffer(self, episode: Dict[str, np.ndarray], eps_len: int) -> None:
        if self._local_replay_buffer is None:
            self._initialize_local_buffer()

        remaining = self._max_size - self._buffer_size
        take = min(int(eps_len), int(remaining))
        if take <= 0:
            return

        start_idx = self._buffer_size
        end_idx = self._buffer_size + take
        episode_idx = len(self._episode_boundaries)
        self._episode_boundaries.append((start_idx, end_idx))
        if take >= self.nstep:
            self._valid_episodes.append(episode_idx)

        obs = np.asarray(episode["observations"][:take], dtype=np.float32).reshape((take, *self.observation_shape))
        next_obs = np.asarray(episode["next_observations"][:take], dtype=np.float32).reshape((take, *self.observation_shape))
        acts = np.asarray(episode["actions"][:take], dtype=np.float32).reshape((take, self.action_dim))
        rews = np.asarray(episode["rewards"][:take], dtype=np.float32).reshape((take,))
        dones = np.asarray(episode["dones"][:take]).astype(np.bool_)

        self._local_replay_buffer["observations"][start_idx:end_idx] = torch.from_numpy(obs).to(self.device, dtype=torch.float32, non_blocking=True)
        self._local_replay_buffer["next_observations"][start_idx:end_idx] = torch.from_numpy(next_obs).to(self.device, dtype=torch.float32, non_blocking=True)
        self._local_replay_buffer["actions"][start_idx:end_idx] = torch.from_numpy(acts).to(self.device, dtype=torch.float32, non_blocking=True)
        self._local_replay_buffer["rewards"][start_idx:end_idx] = torch.from_numpy(rews).to(self.device, dtype=torch.float32, non_blocking=True)
        self._local_replay_buffer["dones"][start_idx:end_idx] = torch.from_numpy(dones).to(self.device, dtype=torch.bool, non_blocking=True)

        if "action_mask" in episode:
            mask = np.asarray(episode["action_mask"][:take], dtype=np.int32).reshape((take,))
            self._local_replay_buffer["action_mask"][start_idx:end_idx] = torch.from_numpy(mask).to(self.device, dtype=torch.int32, non_blocking=True)
        else:
            self._local_replay_buffer["action_mask"][start_idx:end_idx] = 1

        if "clean_action" in episode:
            ca = np.asarray(episode["clean_action"][:take], dtype=np.float32).reshape((take, self.action_dim))
            self._local_replay_buffer["clean_action"][start_idx:end_idx] = torch.from_numpy(ca).to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            self._local_replay_buffer["clean_action"][start_idx:end_idx] = self._local_replay_buffer["actions"][start_idx:end_idx]

        if "global_step" in episode:
            gs = np.asarray(episode["global_step"][:take], dtype=np.int32).reshape((take,))
            self._local_replay_buffer["global_step"][start_idx:end_idx] = torch.from_numpy(gs).to(self.device, dtype=torch.int32, non_blocking=True)
        else:
            self._local_replay_buffer["global_step"][start_idx:end_idx] = 0

        # Pre-compute valid starts for this episode chunk
        if take >= self.nstep:
            valid_starts = torch.arange(start_idx, end_idx - self.nstep + 1, device=self.device, dtype=torch.long)
            self._cache_add_valid_starts_batch(valid_starts)

        self._buffer_size += take

    def _store_current_episode(self, add_to_local_buffer: bool = True) -> None:
        if self._episode_step_count == 0:
            return

        episode: Dict[str, np.ndarray] = {
            "observations": np.asarray(self._current_episode["observations"], dtype=np.float32),
            "next_observations": np.asarray(self._current_episode["next_observations"], dtype=np.float32),
            "actions": np.asarray(self._current_episode["actions"], dtype=np.float32).reshape(-1, self.action_dim),
            "rewards": np.asarray(self._current_episode["rewards"], dtype=np.float32),
            "dones": np.asarray(self._current_episode["dones"], dtype=np.bool_),
        }
        if "action_mask" in self._current_episode:
            episode["action_mask"] = np.asarray(self._current_episode["action_mask"], dtype=np.int32)
        if "clean_action" in self._current_episode:
            episode["clean_action"] = np.asarray(self._current_episode["clean_action"], dtype=np.float32).reshape(-1, self.action_dim)
        if "global_step" in self._current_episode:
            episode["global_step"] = np.asarray(self._current_episode["global_step"], dtype=np.int32)

        eps_len = int(self._episode_step_count)
        if add_to_local_buffer:
            self._add_episode_to_buffer(episode, eps_len)
            self._total_transitions += eps_len

        if self.save_episodes_to_disk:
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            eps_fn = f"{ts}_{eps_len}.npz"
            save_episode(episode, self.replay_dir / eps_fn)

        self._current_episode = defaultdict(list)
        self._episode_step_count = 0

    # -----------------------
    # Sampling
    # -----------------------
    def _sample_uniform_start_indices(self, batch_size: int) -> torch.Tensor:
        count = int(self._valid_starts_count)
        if count <= 0:
            raise RuntimeError(f"No valid {self.nstep}-step windows available for sampling")
        r = torch.randint(0, count, (int(batch_size),), device=self.device)
        return self._valid_starts[:count][r]

    def _build_batch_from_start_indices(self, indices: torch.Tensor) -> Dict[str, Any]:
        if self._local_replay_buffer is None:
            raise RuntimeError("Replay buffer is empty")
        batch_size = int(indices.shape[0])

        # Build step indices for n-step window
        steps = torch.arange(self.nstep, device=self.device, dtype=torch.long)  # (S,)
        step_indices = indices[:, None] + steps[None, :]  # (B, S)

        step_dones = self._local_replay_buffer["dones"][step_indices]  # (B, S)
        done_any = step_dones.any(dim=1)
        first_done_pos = torch.argmax(step_dones.to(torch.int64), dim=1)
        last_offsets = torch.where(done_any, first_done_pos, torch.full_like(first_done_pos, self.nstep - 1))
        effective_indices = indices + last_offsets

        batch: Dict[str, Any] = {}
        batch["observations"] = self._local_replay_buffer["observations"][indices]
        batch["next_observations"] = self._local_replay_buffer["next_observations"][effective_indices]
        batch["actions"] = self._local_replay_buffer["actions"][indices]

        # n-step rewards + discounts (same logic as previous implementation)
        if self.nstep == 1:
            batch["rewards"] = self._local_replay_buffer["rewards"][indices]
            final_dones = self._local_replay_buffer["dones"][indices]
            base_discount = torch.full((batch_size,), self.discount, dtype=torch.float32, device=self.device)
            batch["discounts"] = torch.where(final_dones, torch.zeros_like(base_discount), base_discount)
        else:
            step_rewards = self._local_replay_buffer["rewards"][step_indices]  # (B, S)
            discount_powers = (self.discount ** steps.float()).to(torch.float32)  # (S,)
            not_done = (~step_dones).to(torch.float32)
            prev_not_done = torch.cat(
                [torch.ones((batch_size, 1), device=self.device, dtype=torch.float32), not_done[:, :-1]],
                dim=1,
            )
            alive_mask = torch.cumprod(prev_not_done, dim=1)
            masked_rewards = step_rewards * alive_mask
            n_step_rewards = (masked_rewards * discount_powers[None, :]).sum(dim=1)
            batch["rewards"] = n_step_rewards
            base_discount = torch.full(
                (batch_size,),
                float(self.discount) ** int(self.nstep),
                dtype=torch.float32,
                device=self.device,
            )
            batch["discounts"] = torch.where(done_any, torch.zeros_like(base_discount), base_discount)

        batch["dones"] = self._local_replay_buffer["dones"][effective_indices]
        batch["action_mask"] = self._local_replay_buffer["action_mask"][indices]
        batch["clean_action"] = self._local_replay_buffer["clean_action"][indices]
        batch["global_step"] = self._local_replay_buffer["global_step"][indices]
        return batch

    def sample(self, batch_size: int) -> Dict[str, Any]:
        indices = self._sample_uniform_start_indices(batch_size)
        return self._build_batch_from_start_indices(indices)

    def action_mask_sample(self, batch_size: int, target_action_mask: int, on_policy_ratio: float = 1.0) -> Dict[str, Any]:
        if not (0.0 <= float(on_policy_ratio) <= 1.0):
            raise ValueError(f"on_policy_ratio must be in [0, 1], got {on_policy_ratio}")

        n_on = int(int(batch_size) * float(on_policy_ratio))
        n_off = int(batch_size) - n_on
        if n_on <= 0:
            return self.sample(batch_size)

        target_val = int(target_action_mask)
        self._ensure_action_mask_cache(target_val)

        current_version = int(self._valid_starts_count)
        cached_version = int(self._action_mask_cache_version.get(target_val, -1))
        if cached_version != current_version:
            all_starts = self._valid_starts[:current_version]
            masks = self._local_replay_buffer["action_mask"][all_starts].to(torch.long)
            sel = all_starts[masks == target_val]
            self._action_mask_valid_starts[target_val] = sel
            self._action_mask_valid_starts_count[target_val] = int(sel.numel())
            self._action_mask_cache_version[target_val] = current_version

        buf = self._action_mask_valid_starts[target_val]
        count = int(self._action_mask_valid_starts_count[target_val])
        if count <= 0:
            raise RuntimeError("No valid masked windows available for sampling")

        r = torch.randint(0, count, (n_on,), device=self.device)
        indices_on = buf[:count][r]

        if n_off > 0:
            indices_off = self._sample_uniform_start_indices(n_off)
            indices = torch.cat([indices_on, indices_off], dim=0)
            perm = torch.randperm(indices.shape[0], device=self.device)
            indices = indices[perm]
        else:
            indices = indices_on

        return self._build_batch_from_start_indices(indices)



