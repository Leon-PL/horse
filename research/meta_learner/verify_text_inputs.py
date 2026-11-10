"""Recompute cached text attribution from raw accepted corpus values."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import rtv_text
from src.ingestion.storage import atomic_json
from src.modelling.evaluation import code_signature
from src.modelling.text import prepare_roles


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    root = Path(args.experiment)
    receipt = json.loads((root / "corpus_receipt.json").read_text())
    if sha(root / "corpus.parquet") != receipt["corpus_sha256"]:
        raise ValueError("Text corpus bytes changed")
    frame = pd.read_parquet(root / "corpus.parquet")
    rebuilt = rtv_text.prepare(frame)
    for column in ("plain_name", "verdict_text", "verdict_rank"):
        pd.testing.assert_series_equal(frame[column], rebuilt[column])
    roles = prepare_roles(frame)
    role_path = root / ("roles_" + code_signature()[:16] + ".parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(role_path), roles)
    present = pd.Series(rtv_text.has_text(rebuilt), index=frame.index)
    role_present = roles.model_comment.str.strip().ne("") | roles.model_verdict.str.strip().ne("")
    legacy_fields = present.groupby(frame.race_id).transform("all")
    role_fields = role_present.groupby(frame.race_id).transform("all")
    np.testing.assert_array_equal(legacy_fields.to_numpy(), frame.complete_text.to_numpy())
    result = {**receipt, "roles_sha256": sha(role_path), "code_signature": code_signature(),
              "legacy_preparation_recomputed": True, "role_preparation_recomputed": True,
              "runner_presence_differences": int(present.ne(role_present).sum()),
              "field_presence_difference_races": int(frame.loc[legacy_fields.ne(role_fields), "race_id"].nunique()),
              "duplicate_runners": int(frame.duplicated(["race_id", "horse_id"]).sum())}
    atomic_json(root / "input_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
