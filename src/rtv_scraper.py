"""Compatibility path for :mod:`src.ingestion.racing_tv`.

Collection implementation lives in ``src/ingestion``. Both import paths
resolve to the same module, including patched globals and writer locks.
"""
import sys

from src.ingestion import racing_tv as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
