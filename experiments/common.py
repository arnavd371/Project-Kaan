"""Shared constants for pest-detection benchmarks."""

from __future__ import annotations

CLASS_NAMES = [
    "clean",
    "rice_weevil",
    "lesser_grain_borer",
    "red_flour_beetle",
]

# Cited Balingbing et al. (2024) accuracy used as the workshop reference line.
REFERENCE_PAPER_VAL_ACC = 0.8451

DATA_SUBDIRS = CLASS_NAMES
