"""
Placeholder module for historical forecast ingest.

Phase 1 of the project relies on current forecasts and limited backfill.
This module will host NDFD archive readers (GRIB/Zarr) or a local snapshotter.
"""

from __future__ import annotations

from pathlib import Path


def read_ndfd_archive(path: Path):
    raise NotImplementedError("NDFD archive ingest will be implemented in Phase 2.")


__all__ = ["read_ndfd_archive"]
