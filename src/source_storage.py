"""Compatibility path for :mod:`src.ingestion.storage`.

Collection implementation lives in ``src/ingestion``. Both import paths
resolve to the same module, including patched globals and writer locks.
"""
import sys

from src.ingestion import storage as _implementation

sys.modules[__name__] = _implementation
