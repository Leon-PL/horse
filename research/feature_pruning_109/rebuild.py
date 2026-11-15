"""Full store rebuild pinned to one dataset publication, so before/after fingerprints see the same source data.

    python -m research.feature_pruning_109.rebuild [--manifest ID]
"""
import argparse
import time

BASELINE = "8c61db28718a325efb5b99e609e0e4452bd1bae6ce8cf214d8aa6542a1e2d518"


def main(identifier):
    from src.source_dataset import load_manifest, use_manifest
    from src.store_refresh import get_training_dataset
    started = time.time()
    with use_manifest(load_manifest(identifier)):
        frame = get_training_dataset(rebuild=True)
    print("REBUILD_SECONDS", round(time.time() - started, 1), frame.shape, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=BASELINE)
    main(parser.parse_args().manifest)
