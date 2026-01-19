#!/usr/bin/env python3
"""config.py

Central configuration for the cost-of-living visualization pipeline.

Folder layout:

  <project_root>/
    data/      # input CSV files + DAGI GPKG
    figures/   # output PNGs + combined PDF + debug report
    cache/     # parsed JSON cache

By default, project_root is detected as the directory containing this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    figures_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 220


def detect_root() -> Path:
    """
    Detect project root.

    IMPORTANT: Use the directory containing this file, NOT os.getcwd().
    This avoids bugs where a file path (e.g. .../config.py) is treated as a directory.
    """
    return Path(__file__).resolve().parent


def make_paths(root: Path | None = None) -> Paths:
    root = (root or detect_root()).resolve()
    return Paths(
        root=root,
        data_dir=root / "data",
        figures_dir=root / "figures",
        cache_dir=root / "cache",
    )

