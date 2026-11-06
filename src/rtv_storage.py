"""Compatibility path for :mod:`src.ingestion.rtv_storage`.

Collection implementation lives in ``src/ingestion``. Both import paths
resolve to the same module, including patched globals and writer locks.
"""
import sys

from src.ingestion import rtv_storage as _implementation

sys.modules[__name__] = _implementation
