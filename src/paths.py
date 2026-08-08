"""Canonical project paths, resolved from one place.

Every notebook used to carry its own copy of a root-detection block that keyed on a folder
literally named `data_collection`. That made the folder name load-bearing: thirteen notebooks broke
if it was renamed, and each raised outright if the working directory was anywhere unexpected.

This resolves the root by looking for `pytest.ini`, which sits at the repo root and is not tied to
any layout decision, so directories can be reorganised again without touching a single notebook.
"""
from __future__ import annotations

from pathlib import Path

_MARKERS = ("pytest.ini", ".git")


def repo_root(start: Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if any((candidate / m).exists() for m in _MARKERS):
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root: no pytest.ini or .git found in any parent of "
        f"{here}. Open the notebook from inside the repository."
    )


REPO_ROOT = repo_root(Path(__file__).parent)
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = REPO_ROOT / "outputs"
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

for _d in (DATA_DIR, RAW_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
