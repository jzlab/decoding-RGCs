"""Shared pytest configuration for the spectral fill-in test suite."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

ARTIFACT_DIR = REPO_ROOT / "results_prior-vs-likelihood" / "tests"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
