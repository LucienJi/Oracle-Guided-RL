#!/usr/bin/env python3
"""
Plot saved visualization data from .npy files.
Loads pre-computed IQM results and plots them without reprocessing raw data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import hydra
from omegaconf import DictConfig, OmegaConf


def load_plot_data(npy_path: Path) -> Dict[str, Any]:
    """Load saved plot data from .npy file."""
    if not npy_path.exists():
        raise FileNotFoundError(f"Data file not found: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def plot_from_saved(config: Dict) -> None:
    """
    Plot visualization from saved .npy data file using config.
    
    Args:
        config: Configuration dictionary with NPY_PATH and PLOT_SETTINGS
    """
    npy_path = Path(config["NPY_PATH"])
    plot_cfg = config["PLOT_SETTINGS"]
    
    # Load saved data
    data = load_plot_data(npy_path)
    x_grid = data["x_grid"]
    iqm_results = data["iqm_results"]
    
    # Extract plot settings
    base_font = plot_cfg.get("font_size", 16)
    legend_font = plot_cfg.get("legend_font_size", 14)
    show_legend = plot_cfg.get("show_legend", True)
    figsize = plot_cfg.get("figsize", [8, 5])
    dpi = plot_cfg.get("dpi", 200)
    
    # Setup plotting style
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
    
    # Create plot
    plt.figure(figsize=figsize)
    for algo, algo_data in iqm_results.items():
        iqm = algo_data["iqm"]
        lower = algo_data["lower"]
        upper = algo_data["upper"]
        plt.plot(x_grid, iqm, label=algo)
        plt.fill_between(x_grid, lower, upper, alpha=0.2)
    
    # Set labels and title
    plt.xlabel(plot_cfg.get("x_label", "Environment Steps"))
    plt.ylabel(plot_cfg.get("y_label", "Score"))
    if plot_cfg.get("title") is not None:
        plt.title(plot_cfg["title"])
    
    # Set x-axis to scientific notation to avoid crowded labels
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    
    if show_legend:
        plt.legend()
    plt.tight_layout()
    
    # Save if path provided
    if config.get("SAVE_PATH"):
        save_path = Path(config["SAVE_PATH"])
        plt.savefig(save_path, dpi=dpi)
        print(f"Plot saved to: {save_path}")
    
    plt.show()


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "viz_config"),
    config_name="plot_saved",
)
def main(cfg: DictConfig) -> None:
    """Main entry point using Hydra configuration."""
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Validate required config keys
    if "NPY_PATH" not in config:
        raise KeyError("Missing required config key: NPY_PATH")
    if "PLOT_SETTINGS" not in config:
        raise KeyError("Missing required config key: PLOT_SETTINGS")
    
    plot_from_saved(config)


if __name__ == "__main__":
    main()

