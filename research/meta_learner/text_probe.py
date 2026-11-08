"""A registered retrospective routing test: generalist plus stored text scores.

No base model is refitted. Earlier generalist OOF forecasts fit a two-input
pool; a purged later part of the same quarter is scored. The generalist already
uses text features, so this tests an alternative combination, not new information.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.storage import atomic_json, atomic_parquet, writer_lock
from src.modelling.data import ordered_frame
from src.modelling.evaluation import code_signature, nested_pool_evaluation
from src.race_objectives import race_bounds, race_softmax
from src.wf_eval import Fold


def benchmark_fields(store_root, registered, base):
    """Reuse the completed benchmark's checked generation, not collector inputs.

    Only scoring context and one existing stored score are read. No processing
    or new feature calculation is needed to inspect the same immutable build.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.feature_store import FeatureStore
    from src.featured_dataset import fe_signature
    from src.source_dataset import load_manifest
    receipt = registered["receipt"]
    store = FeatureStore(store_root)
    with store._access():
        meta = store.meta()
        if (meta.get("fe_signature") != receipt["fe_signature"] or fe_signature() != receipt["fe_signature"]
                or meta.get("data_signature") != receipt["data_signature"]
                or (meta.get("dataset_manifest") or {}).get("id") != receipt["dataset_manifest"]["id"]
                or meta.get("last_date") != receipt["store_last_date"]):
            raise ValueError("Store generation differs from the completed benchmark")
        manifest = meta["dataset_manifest"]
        load_manifest(manifest["id"], root=manifest.get("root"))
        schema = pq.read_schema(store.featured_path)
        dates = [pd.Timestamp(base.race_date.min()), pd.Timestamp(base.race_date.max())]
        if pa.types.is_string(schema.field("race_date").type):
            dates = [str(day.date()) for day in dates]
        columns = ["race_id", "horse_id", "horse_name", "race_date", "num_runners", "won", "finish_position", "rtv_text_logit"]
        frame = pd.read_parquet(store.featured_path, columns=columns,
            filters=[("race_date", ">=", dates[0]), ("race_date", "<=", dates[1])])
    frame = frame.loc[frame.race_id.isin(base.race_id)].reset_index(drop=True)
    frame, _ = ordered_frame(frame)
    return frame, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--train-through", required=True)
    parser.add_argument("--score-from", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    folder, output = Path(args.benchmark), Path(args.output)
    registered = json.loads((folder / "registration.json").read_text())
    completed = json.loads((folder / "report.json").read_text())
    if registered["code_signature"] != code_signature():
        raise ValueError("Benchmark modelling code changed")
    if hashlib.sha256((folder / "oof.parquet").read_bytes()).hexdigest() != completed["oof_sha256"]:
        raise ValueError("Benchmark predictions changed")
    if len(registered["specs"]) != 1:
        raise ValueError("The routing probe requires one locked generalist baseline")
    base = pd.read_parquet(folder / "oof.parquet")
    generalist = registered["specs"][0]["id"]
    base = base.loc[base.component.eq(generalist)].copy()
    frame, receipt = benchmark_fields(args.store_root, registered, base)
    if receipt["dataset_manifest"]["id"] != registered["receipt"]["dataset_manifest"]["id"]:
        raise ValueError("Text and generalist must use the same accepted dataset")
    gap = registered["specs"][0]["purge_days"]
    last, first = pd.Timestamp(args.train_through), pd.Timestamp(args.score_from)
    if first <= last + pd.Timedelta(days=gap):
        raise ValueError("Pool selection and scoring need a chronological purge")
    registration = {"benchmark": str(folder.resolve()), "base_oof_sha256": completed["oof_sha256"],
        "dataset": receipt["dataset_manifest"]["id"], "fe_signature": receipt["fe_signature"],
        "train_through": str(last.date()), "score_from": str(first.date()), "purge_days": gap,
        "shrinkage": .01, "code_signature": code_signature(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Retrospective combination test; generalist already consumes the stored text features"}
    output.mkdir(parents=True, exist_ok=True)
    with writer_lock(output / ".probe.lock") as acquired:
        if not acquired:
            raise RuntimeError("Text probe is already running")
        path = output / "registration.json"
        if path.exists() and json.loads(path.read_text()) != registration:
            raise ValueError("Probe output belongs to a different registration")
        atomic_json(path, registration)
        present = frame.rtv_text_logit.notna().groupby(frame.race_id).transform("all").to_numpy()
        # Missing or partial text fields are unavailable as complete experts.
        p = race_softmax(frame.rtv_text_logit.fillna(0).to_numpy(), race_bounds(frame.race_id))
        p[~present] = np.nan
        text = frame[["race_id", "horse_id", "race_date"]].copy()
        text["component"], text["probability"], text["available"] = "stored-timeform-text", p, present
        records = pd.concat([base, text], ignore_index=True)
        fold = Fold("later-quarter-text-probe", frame.race_date.le(last).to_numpy(),
            frame.race_date.ge(first).to_numpy(), int(frame.race_date.between(last, first, inclusive="neither").sum()))
        report, predicted, races = nested_pool_evaluation(frame, records,
            (generalist, "stored-timeform-text"), folds=[fold], purge_days=gap, shrinkage=.01)
        report["text_complete_races"] = int(frame.loc[present, "race_id"].nunique())
        report["registration"] = registration
        atomic_parquet(predicted, output / "predictions.parquet")
        atomic_parquet(races, output / "races.parquet")
        atomic_json(output / "report.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
