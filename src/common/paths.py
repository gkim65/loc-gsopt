"""
Repository-relative paths, so scripts behave the same from any working directory.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURES_DIR = os.path.join(REPO_ROOT, "figures_final")


def repo_path(relative):
    """
        repo_path(relative) -> str

    Resolve a path given relative to the repository root.

      - `relative` — path from the repository root, e.g. "data/city_buffers.shp"

    Returns the absolute path.
    """
    return os.path.join(REPO_ROOT, relative)


def figure_path(name):
    """
        figure_path(name) -> str

    Path for a figure output, creating `FIGURES_DIR` if it does not exist yet.

      - `name` — file name inside `FIGURES_DIR`, e.g. "heatmap_comparison.pdf"

    Returns the absolute path.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)
