"""Verify saved text forecasts/artifacts and report calibration on matched races."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_selection import race_groups
from src.forecast_evaluation import probability_metrics, paired_interval
from src.ingestion.storage import atomic_json
from src.modelling.evaluation import code_signature


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reliability(frame):
    bins = pd.cut(frame.probability, [-np.inf, .01, .025, .05, .1, .2, .3, .5, 1.], right=True)
    rows = []
    for label, part in frame.groupby(bins, observed=True):
        rows.append({"band": str(label), "runners": len(part),
                     "predicted": float(part.probability.mean()), "observed": float(part.won.mean())})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    root = Path(args.experiment)
    selection = json.loads((root / "selection.json").read_text())
    confirmation = json.loads((root / "confirmation.json").read_text())
    design = json.loads((root / "pre_registration.json").read_text())
    receipt = json.loads((root / "corpus_receipt.json").read_text())
    identity = selection["identity"]
    if confirmation["identity"] != identity or code_signature() != identity["code"]:
        raise ValueError("Verification requires the evaluated source version")
    if sha(root / "corpus.parquet") != receipt["corpus_sha256"]:
        raise ValueError("Evaluation corpus bytes changed")
    corpus = pd.read_parquet(root / "corpus.parquet")
    corpus = corpus.loc[corpus.complete_text].reset_index(drop=True)
    arms = list(dict.fromkeys(["corrected", selection["selected"]]))
    reports, losses, artifacts = {}, {}, []
    for arm in arms:
        settings = design["candidates"][arm]
        key = hashlib.sha256(json.dumps({**identity, "settings": settings}, sort_keys=True).encode()).hexdigest()
        frame = pd.read_parquet(root / f"confirm_{arm}.parquet")
        quarters = {}
        for label, part in frame.groupby("quarter", sort=True):
            cache = root / "cache" / key / label
            stored = json.loads((cache / "receipt.json").read_text())
            if stored["identity"] != identity or stored["settings"] != settings:
                raise ValueError("Forecast receipt differs from the locked recipe")
            if stored["sha256"] != sha(cache / "predictions.parquet"):
                raise ValueError("Forecast bytes changed")
            test = corpus.loc[corpus.race_date.dt.to_period("Q").astype(str).eq(label)].copy()
            if not np.array_equal(test[["race_id", "horse_id"]].to_numpy(), part[["race_id", "horse_id"]].to_numpy()):
                raise ValueError("Scored fields do not match the accepted corpus")
            boundary = test.race_date.min() - pd.Timedelta(days=design["purge_days"])
            lineage = stored["lineage"]
            if (pd.Timestamp(lineage["training_last_date"]) > boundary
                    or pd.Timestamp(lineage["calibration_last_date"]) > boundary
                    or pd.Timestamp(lineage["calibration_fit_boundary"]) > pd.Timestamp(lineage["calibration_first_date"]) - pd.Timedelta(days=7)):
                raise ValueError("Artifact training/calibration violated the purge")
            model = joblib.load(cache / "model.joblib")
            inference = test.drop(columns=["won", "finish_position", "bsp", "odds"], errors="ignore")
            replay = model.predict(inference)
            np.testing.assert_allclose(replay, part.probability, rtol=0, atol=1e-12)
            sample = inference.loc[inference.race_date.eq(inference.race_date.min())].copy()
            expected = model.predict(sample)
            shuffled = sample.sample(frac=1, random_state=42)
            lookup = pd.Series(expected, index=sample.index)
            np.testing.assert_allclose(model.predict(shuffled), lookup.loc[shuffled.index], rtol=0, atol=1e-12)
            changed = sample.copy()
            changed[["won", "finish_position", "bsp", "odds"]] = 999.
            np.testing.assert_allclose(model.predict(changed), expected, rtol=0, atol=0)
            absent = sample.copy()
            absent.iloc[0, absent.columns.get_indexer(["tf_comment", "tf_verdict"])] = None
            missing = model.predict(absent)
            unavailable = absent.race_id.eq(absent.race_id.iloc[0]).to_numpy()
            assert np.isnan(missing[unavailable]).all()
            np.testing.assert_allclose(missing[~unavailable], expected[~unavailable], rtol=0, atol=0)
            metric, loss, _ = probability_metrics(part.probability, part.won, race_groups(part.race_id))
            quarters[label] = {**metric, "races": int(part.race_id.nunique()), "scale": lineage["scale"]}
            artifacts.append({"arm": arm, "quarter": label, "path": str((cache / "model.joblib").resolve()),
                              "sha256": sha(cache / "model.joblib"), "forecast_replay_max_error": float(np.max(np.abs(replay - part.probability))),
                              "target_invariance": True, "shuffled_alignment": True, "whole_field_missingness": True})
            part = part.copy()
            part["verified_loss"] = np.repeat(loss, race_groups(part.race_id))
            losses.setdefault(arm, []).append(part.drop_duplicates("race_id").set_index("race_id")[["race_date", "verified_loss"]])
        metric, _, _ = probability_metrics(frame.probability, frame.won, race_groups(frame.race_id))
        reports[arm] = {**metric, "runners": len(frame), "races": int(frame.race_id.nunique()),
                        "quarters": quarters, "reliability": reliability(frame)}
        losses[arm] = pd.concat(losses[arm])
    reference = losses["corrected"]
    differences = {}
    for arm in arms:
        if arm != "corrected":
            candidate = losses[arm].loc[reference.index]
            delta = candidate.verified_loss.to_numpy() - reference.verified_loss.to_numpy()
            differences[arm] = {"date_cluster": paired_interval(delta, blocks=reference.race_date),
                                "race_bootstrap": paired_interval(delta)}
    result = {"identity": identity, "selection_sha256": sha(root / "selection.json"),
              "confirmation_sha256": sha(root / "confirmation.json"), "reports": reports,
              "paired_differences": differences, "artifacts": artifacts,
              "evaluation_status": design["evaluation_status"]}
    atomic_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
