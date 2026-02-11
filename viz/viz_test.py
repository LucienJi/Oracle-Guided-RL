#!/usr/bin/env python3
"""
Highly modular RL results visualization using rliable.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import hydra
from omegaconf import DictConfig, OmegaConf

from rliable import library as rly


# =========================
# CONFIG (see YAML in viz_config/)
# =========================


def _validate_config(config: Dict) -> None:
    required_top = [
        "ROOT_DIR",
        "FILE_PATTERN",
        "COLUMN_MAPPING",
        "FILTERS",
        "NORMALIZATION",
        "PLOT_SETTINGS",
    ]
    for key in required_top:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    if "step" not in config["COLUMN_MAPPING"].values():
        raise ValueError("COLUMN_MAPPING must map one column to internal name 'step'.")
    if "score" not in config["COLUMN_MAPPING"].values():
        raise ValueError("COLUMN_MAPPING must map one column to internal name 'score'.")

    if config["USE_SUBFOLDERS"]:
        if config["ALGO_DIRS"]:
            raise ValueError("When USE_SUBFOLDERS=True, ALGO_DIRS must be empty.")
    else:
        if not config["ALGO_DIRS"]:
            raise ValueError("When USE_SUBFOLDERS=False, ALGO_DIRS must be provided.")
    if config["ALGO_NAMES"] and config["ALGO_DIRS"] and len(config["ALGO_NAMES"]) != len(
        config["ALGO_DIRS"]
    ):
        raise ValueError("ALGO_NAMES must match length of ALGO_DIRS when provided.")


def _find_eval_files(algo_dir: Path, pattern: str) -> List[Path]:
    matcher = re.compile(pattern)
    return sorted([p for p in algo_dir.iterdir() if p.is_file() and matcher.match(p.name)])


def _rename_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    missing = [src for src in column_mapping.keys() if src not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")
    return df.rename(columns=column_mapping)



def _cache_settings(config: Dict) -> Tuple[bool, Path]:
    cache_cfg = config.get("CACHE", {})
    enabled = bool(cache_cfg.get("enabled", False))
    cache_dir = Path(cache_cfg.get("dir", "viz_cache"))
    if not cache_dir.is_absolute():
        cache_dir = Path(config["ROOT_DIR"]) / cache_dir
    return enabled, cache_dir


def _build_cache_key(eval_files: List[Path], config: Dict) -> str:
    # Cache key includes file metadata and data processing parameters.
    file_meta = [
        {
            "name": p.name,
            "mtime": p.stat().st_mtime,
            "size": p.stat().st_size,
        }
        for p in eval_files
    ]
    payload = {
        "file_pattern": config["FILE_PATTERN"],
        "column_mapping": config["COLUMN_MAPPING"],
        "filters": config["FILTERS"],
        "interpolation": config.get("INTERPOLATION", {}),
        "files": file_meta,
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def _load_cache(cache_dir: Path, algo_name: str, cache_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    cache_path = cache_dir / f"{algo_name}_{cache_key}.npz"
    if not cache_path.exists():
        return None
    cached = np.load(cache_path)
    original_steps = cached.get("original_steps", None)
    return cached["x_grid"], cached["scores"], original_steps


def _save_cache(cache_dir: Path, algo_name: str, cache_key: str, x_grid: np.ndarray, scores: np.ndarray, original_steps: np.ndarray | None = None) -> None:
    cache_path = cache_dir / f"{algo_name}_{cache_key}.npz"
    if original_steps is not None:
        np.savez_compressed(cache_path, x_grid=x_grid, scores=scores, original_steps=original_steps)
    else:
        np.savez_compressed(cache_path, x_grid=x_grid, scores=scores)


def _read_seed_csv(args: Tuple[Path, Dict[str, str], Dict[str, float], int]) -> pd.DataFrame:
    csv_path, column_mapping, filters, seed_id = args
    df = pd.read_csv(csv_path)
    df = _rename_columns(df, column_mapping)
    df = df[["step", "score"]]
    df = df[(df["step"] >= filters["min_step"]) & (df["step"] <= filters["max_step"])]
    if df.empty:
        raise ValueError(f"No data within filter range for {csv_path}")
    df = df.assign(seed_id=seed_id)
    return df


def load_and_aggregate(algo_dir: Path, config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load all seed CSVs for a given algorithm and return:
    - x_grid: common x-axis for interpolation
    - scores: array shape (num_seeds, len(x_grid))
    - original_steps: array of original data step positions (None if interpolation disabled)
    """
    eval_files = _find_eval_files(algo_dir, config["FILE_PATTERN"])
    if not eval_files:
        raise FileNotFoundError(f"No eval files found in {algo_dir} with {config['FILE_PATTERN']}")

    filters = config["FILTERS"]
    interpolation_enabled = config.get("INTERPOLATION", {}).get("enabled", True)
    parallel_cfg = config.get("PARALLEL", {})
    parallel_enabled = bool(parallel_cfg.get("enabled", False))
    max_workers = parallel_cfg.get("max_workers", None)

    cache_enabled, cache_dir = _cache_settings(config)
    cache_key = _build_cache_key(eval_files, config)
    if cache_enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = _load_cache(cache_dir, algo_dir.name, cache_key)
        if cached is not None:
            return cached

    # Batch read all CSVs and do a single groupby across seeds and steps.
    read_args = [(csv_path, config["COLUMN_MAPPING"], filters, idx) for idx, csv_path in enumerate(eval_files)]
    if parallel_enabled:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            dfs = list(executor.map(_read_seed_csv, read_args))
    else:
        dfs = [_read_seed_csv(args) for args in read_args]

    combined = pd.concat(dfs, ignore_index=True)
    grouped = (
        combined.groupby(["seed_id", "step"], as_index=False)["score"].mean().sort_values(["seed_id", "step"])
    )

    num_seeds = len(eval_files)
    x_grid = None
    if interpolation_enabled:
        x_grid = np.linspace(filters["min_step"], filters["max_step"], filters["grid_points"])
        # Preallocate to avoid repeated list growth and vstack copies.
        scores = np.full((num_seeds, len(x_grid)), np.nan, dtype=float)
        # Collect all original step positions across all seeds
        all_original_steps = set()
        for seed_id in range(num_seeds):
            seed_df = grouped[grouped["seed_id"] == seed_id]
            if seed_df.empty:
                raise ValueError(f"No data within filter range for seed index {seed_id}")
            seed_df = seed_df.sort_values("step")
            all_original_steps.update(seed_df["step"].values)
            scores[seed_id] = np.interp(
                x_grid, seed_df["step"].values, seed_df["score"].values, left=np.nan, right=np.nan
            )
        original_steps = np.array(sorted(all_original_steps))
        if cache_enabled:
            _save_cache(cache_dir, algo_dir.name, cache_key, x_grid, scores, original_steps)
        return x_grid, scores, original_steps

    # No interpolation: require perfectly aligned steps
    base_steps = None
    scores = None
    for seed_id in range(num_seeds):
        seed_df = grouped[grouped["seed_id"] == seed_id]
        if seed_df.empty:
            raise ValueError(f"No data within filter range for seed index {seed_id}")
        seed_steps = seed_df["step"].values
        seed_scores = seed_df["score"].values
        if base_steps is None:
            base_steps = seed_steps
            scores = np.empty((num_seeds, len(base_steps)), dtype=float)
        if not np.array_equal(base_steps, seed_steps):
            raise ValueError(
                "Interpolation disabled, but seed steps are not aligned "
                f"(mismatch at seed index {seed_id})."
            )
        scores[seed_id] = seed_scores
    x_grid = base_steps
    if cache_enabled:
        _save_cache(cache_dir, algo_dir.name, cache_key, x_grid, scores)
    # When interpolation is disabled, all steps are original data points
    return x_grid, scores, None


def _apply_normalization(scores: np.ndarray, config: Dict) -> np.ndarray:
    if not config["NORMALIZATION"]["enabled"]:
        return scores
    baseline = config["NORMALIZATION"]["baseline_score"]
    return scores / baseline if baseline != 0 else scores


def _rliable_iqm_ci(
    scores_dict: Dict[str, np.ndarray], num_bootstrap: int = 100
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute IQM and bootstrap CIs across seeds for each x.
    Returns dict: algo -> (iqm, (lower, upper)).
    """
    def metric_fn(x: np.ndarray) -> np.ndarray:
        # Vectorized IQM across tasks to avoid per-step Python loops.
        with np.errstate(invalid="ignore"):
            q1 = np.nanquantile(x, 0.25, axis=0)
            q3 = np.nanquantile(x, 0.75, axis=0)
            mask = (x >= q1) & (x <= q3)
            return np.nanmean(np.where(mask, x, np.nan), axis=0)

    iqm_scores, iqm_cis = rly.get_interval_estimates(scores_dict, metric_fn, reps=num_bootstrap)
    out = {}
    for algo, iqm in iqm_scores.items():
        lower = iqm_cis[algo][0]
        upper = iqm_cis[algo][1]
        out[algo] = (iqm, (lower, upper))
    return out


def plot_algorithms(
    x_grid: np.ndarray,
    scores_by_algo: Dict[str, np.ndarray],
    config: Dict,
    original_steps: np.ndarray | None = None,
) -> None:
    plot_cfg = config["PLOT_SETTINGS"]
    base_font = plot_cfg.get("font_size", 12)
    legend_font = plot_cfg.get("legend_font_size", base_font)
    show_legend = plot_cfg.get("show_legend", True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": base_font,
            "axes.titlesize": base_font,
            "axes.labelsize": base_font,
            "xtick.labelsize": base_font,
            "ytick.labelsize": base_font,
            "legend.fontsize": legend_font,
        }
    )

    normalized = {
        algo: _apply_normalization(scores, config) for algo, scores in scores_by_algo.items()
    }
    iqm_results = _rliable_iqm_ci(normalized)

    plt.figure(figsize=config["PLOT_SETTINGS"]["figsize"])
    # Create mask for valid data regions if original_steps is provided (interpolation enabled)
    if original_steps is not None and len(original_steps) > 0:
        # Only fill between points that are within the range of consecutive original data points
        # This ensures we don't fill regions where there's no actual data
        sorted_steps = np.sort(original_steps)
        valid_mask = np.zeros(len(x_grid), dtype=bool)
        # Mark points between consecutive original data points
        for i in range(len(sorted_steps) - 1):
            mask = (x_grid >= sorted_steps[i]) & (x_grid <= sorted_steps[i + 1])
            valid_mask |= mask
        # Also include points very close to original data points (within small tolerance)
        tolerance = np.median(np.diff(sorted_steps)) * 0.01 if len(sorted_steps) > 1 else 1.0
        for orig_step in sorted_steps:
            valid_mask |= np.abs(x_grid - orig_step) <= tolerance
    else:
        # No interpolation: all points are valid (they are all original data points)
        valid_mask = np.ones(len(x_grid), dtype=bool)
    
    for algo, (iqm, (lower, upper)) in iqm_results.items():
        plt.plot(x_grid, iqm, label=algo)
        # Only fill between points where we have valid data (between consecutive original data points)
        plt.fill_between(x_grid, lower, upper, alpha=0.2, where=valid_mask)

    if config["NORMALIZATION"]["enabled"]:
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="baseline")

    plt.xlabel(plot_cfg["x_label"])
    plt.ylabel(plot_cfg["y_label"])
    if plot_cfg.get("title") is not None:
        plt.title(plot_cfg["title"])
    
    # Set x-axis to scientific notation to avoid crowded labels.
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    
    if show_legend:
        plt.legend()
    plt.tight_layout()
    
    # Save processed data as npy file if SAVE_PATH is provided.
    if config["SAVE_PATH"]:
        save_path = Path(config["SAVE_PATH"])
        plt.savefig(save_path, dpi=200)
        # Save data with same name as image but .npy extension.
        data_path = save_path.with_suffix(".npy")
        plot_data = {
            "x_grid": x_grid,
            "iqm_results": {algo: {"iqm": iqm, "lower": lower, "upper": upper} 
                           for algo, (iqm, (lower, upper)) in iqm_results.items()},
            "scores_by_algo": normalized,
        }
        np.save(data_path, plot_data, allow_pickle=True)
    
    plt.show()


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "viz_config"),
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    _validate_config(config)
    root_dir = Path(config["ROOT_DIR"])

    if config["USE_SUBFOLDERS"]:
        algo_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    else:
        algo_dirs = [
            (Path(p) if Path(p).is_absolute() else root_dir / p) for p in config["ALGO_DIRS"]
        ]

    if not algo_dirs:
        raise FileNotFoundError("No algorithm directories found with the current configuration.")

    algo_names = config["ALGO_NAMES"] or [p.name for p in algo_dirs]
    if len(algo_names) != len(algo_dirs):
        raise ValueError("ALGO_NAMES must match the number of algorithm folders.")

    scores_by_algo: Dict[str, np.ndarray] = {}
    x_grid: np.ndarray | None = None
    original_steps: np.ndarray | None = None
    for algo_dir, algo_name in zip(algo_dirs, algo_names):
        grid, scores, orig_steps = load_and_aggregate(algo_dir, config)
        scores_by_algo[algo_name] = scores
        if x_grid is None:
            x_grid = grid
        if original_steps is None:
            original_steps = orig_steps

    if x_grid is None:
        raise RuntimeError("Failed to build x-axis grid from data.")

    plot_algorithms(x_grid, scores_by_algo, config, original_steps)


if __name__ == "__main__":
    main()

