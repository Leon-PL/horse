"""Refit a locked, evaluated text recipe through an explicit settled cutoff.

Reads accepted raw observations directly; no feature-store rebuild, collection,
source publication or selected-model change is part of this candidate fit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.forecast_evaluation import valid_race_mask
from src.ingestion.storage import atomic_json, writer_lock
from src.modelling.data import ordered_frame
from src.modelling.evaluation import code_signature, frame_signature
from src.modelling.text import TextExpert
from src.source_dataset import load_manifest, use_manifest, source_path, model_table
from src.source_prices import attach_bsp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_input(study, cutoff, receipt):
    """Read the accepted refit input and reconcile it with the scored history."""
    if sha(study / "corpus.parquet") != receipt["corpus_sha256"]:
        raise ValueError("Evaluated canonical text corpus changed")
    manifest = load_manifest(receipt["dataset_id"])
    with use_manifest(manifest):
        path = source_path("sl", "results")
        required = ["race_id", "horse_id", "horse_name", "race_date", "num_runners", "finish_position"]
        schema = pq.read_schema(path).names
        if set(required) - set(schema):
            raise ValueError("Accepted canonical source lacks required field identities")
        frame = pd.read_parquet(path, columns=required + (["odds"] if "odds" in schema else []))
        frame["race_id"] = frame.race_id.astype(str)
        frame["race_date"] = pd.to_datetime(frame.race_date, errors="coerce").dt.normalize()
        # Match clean_data's canonical spelling before verdict attribution.
        frame["horse_name"] = frame.horse_name.astype("string").str.strip().str.title()
        frame = frame.loc[frame.race_date.le(cutoff)].copy()
        frame["finish_position"] = pd.to_numeric(frame.finish_position, errors="coerce")
        frame["won"] = frame.finish_position.eq(1).astype(int)
        if "odds" in frame:
            frame["odds"] = pd.to_numeric(frame.odds, errors="coerce")
        before = frame.race_id.nunique()
        valid = valid_race_mask(frame)
        if "odds" in frame:
            bad = frame.loc[frame.odds.lt(1.01), "race_id"]
            valid &= ~frame.race_id.isin(bad)
        frame = frame.loc[valid].copy()
        text = model_table("rtv", "text")[["sl_race_id", "sl_horse_id", "tf_comment", "tf_verdict"]].rename(
            columns={"sl_race_id": "race_id", "sl_horse_id": "horse_id"})
        text["race_id"], text["horse_id"] = text.race_id.astype(str), pd.to_numeric(text.horse_id)
        frame = frame.merge(text, on=["race_id", "horse_id"], how="left", validate="one_to_one")
        frame = attach_bsp(frame, cols=("bsp",), overwrite=True)
    frame, _ = ordered_frame(frame)
    columns = ["race_id", "horse_id", "horse_name", "race_date", "num_runners",
               "won", "finish_position", "odds", "bsp", "tf_comment", "tf_verdict"]
    reference = pd.read_parquet(study / "corpus.parquet", columns=columns)
    prefix = frame.loc[frame.race_date.le(reference.race_date.max()), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, reference, check_dtype=False, check_exact=True)
    return frame, manifest["id"], int(before - frame.race_id.nunique())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    study, output = Path(args.experiment), Path(args.output)
    cutoff = pd.Timestamp(args.as_of)
    if pd.isna(cutoff) or cutoff != cutoff.normalize() or cutoff >= pd.Timestamp.now().normalize():
        raise ValueError("Candidate cutoff must be a settled calendar date before today")
    selected = json.loads((study / "selection.json").read_text())
    confirmation = json.loads((study / "confirmation.json").read_text())
    if selected["identity"] != confirmation["identity"] or selected["identity"]["code"] != code_signature():
        raise ValueError("Candidate needs an evaluated recipe from the same model mathematics")
    settings = json.loads((study / "selected_text_spec.json").read_text())
    if settings != selected["selected_settings"]:
        raise ValueError("Selected text recipe changed after evaluation")
    receipt = json.loads((study / "corpus_receipt.json").read_text())
    if selected["identity"]["corpus"] != receipt["corpus_sha256"]:
        raise ValueError("Recipe does not match the accepted corpus receipt")
    frame, dataset_id, excluded_races = candidate_input(study, cutoff, receipt)
    input_hash = frame_signature(frame, frame.columns)
    output.mkdir(parents=True, exist_ok=True)
    with writer_lock(output / ".fit.lock") as acquired:
        if not acquired or (output / "model.joblib").exists():
            raise ValueError("Candidate output already exists or is owned by another fit")
        settings["stage_weights"] = tuple(settings["stage_weights"])
        source = code_signature()
        model = TextExpert(id="timeform-text-candidate", **settings).fit(frame)
        model.lineage.update(dataset_id=dataset_id, as_of=str(cutoff.date()), code_signature=source)
        sample = frame.loc[frame.race_date.ge(frame.race_date.max() - pd.Timedelta(days=7))].drop(columns=["won", "finish_position", "bsp", "odds"], errors="ignore")
        expected = model.predict(sample)
        temporary = output / "model.tmp.joblib"
        joblib.dump(model, temporary, compress=3)
        reloaded = joblib.load(temporary)
        np.testing.assert_allclose(reloaded.predict(sample), expected, rtol=0, atol=0, equal_nan=True)
        if code_signature() != source:
            raise RuntimeError("Model code changed during the candidate fit")
        os.replace(temporary, output / "model.joblib")
        metadata = {"model_type": "text_expert_candidate", "as_of": str(cutoff.date()), "dataset_id": dataset_id,
            "historical_prefix_reconciled": True, "corpus_sha256": receipt["corpus_sha256"],
            "input_signature": input_hash, "excluded_invalid_races": excluded_races,
            "code_signature": source, "preparation_script_sha256": sha(__file__), "model_sha256": sha(output / "model.joblib"),
            "selection_sha256": sha(study / "selection.json"), "confirmation_sha256": sha(study / "confirmation.json"),
            "lineage": model.lineage, "reload_verified": True, "serving_qualified": False,
            "performance_claim": "Refitted candidate has no held-out score; refer to the earlier chronological recipe evaluation"}
        atomic_json(output / "metadata.json", metadata)
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
