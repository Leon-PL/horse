"""Lean candidate artifacts in the existing run registry, without promotion."""
from dataclasses import dataclass
import os
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from src.modelling.data import ordered_frame
from src.modelling.evaluation import code_signature, prediction_matrix
from src.modelling.pool import LogPool
from src.modelling.components import component_model


@dataclass
class WinCandidate:
    models: dict
    pool: LogPool
    lineage: dict

    def predict(self, frame):
        if tuple(self.models) != self.pool.components:
            raise ValueError("Saved component order differs from the combiner")
        matrix = np.column_stack([model.predict(frame) for model in self.models.values()])
        return self.pool.predict(matrix, frame.race_id)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            joblib.dump({"format": "win_stack_candidate", "version": 1, "candidate": self}, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        contents = joblib.load(path)
        if contents.get("format") != "win_stack_candidate" or contents.get("version") != 1:
            raise ValueError("Unsupported win candidate artifact")
        result = contents["candidate"]
        if not isinstance(result, cls) or tuple(result.models) != result.pool.components:
            raise ValueError("Invalid candidate component registry")
        return result


def fit_candidate(frame, specs, records, *, as_of, dataset_start=None, capacity="native", threads=1, shrinkage=.01):
    source = code_signature()
    frame, _ = ordered_frame(frame)
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff != cutoff.normalize() or frame.race_date.gt(cutoff).any():
        raise ValueError("Candidate fit must use only rows through its explicit calendar cutoff")
    names = tuple(s.id for s in specs)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate component IDs")
    for spec in specs:
        part = records.loc[records.component.eq(spec.id)]
        if (part.empty or part.recipe_signature.ne(spec.signature).any()
                or pd.to_datetime(part.race_date).gt(cutoff).any()
                or (pd.to_datetime(part.training_last_date) >= pd.to_datetime(part.race_date)).any()):
            raise ValueError("Candidate pool requires earlier OOF predictions from the same recipes")
    matrix = prediction_matrix(frame, records, names)
    usable = pd.Series(np.isfinite(matrix[:, 0])).groupby(frame.race_id).transform("all").to_numpy()
    if not usable.any():
        raise ValueError("No generalist OOF fields to fit the combiner")
    pool = LogPool(names, shrinkage=shrinkage).fit(matrix[usable], frame.won.to_numpy()[usable], frame.race_id.to_numpy()[usable])
    models = {s.id: component_model(s).fit(frame, dataset_start=dataset_start, capacity=capacity, threads=threads) for s in specs}
    if code_signature() != source:
        raise RuntimeError("Modelling code changed during fitting; refusing to publish this candidate")
    lineage = {"as_of": str(cutoff.date()), "code_signature": source,
        "pool_fit_first_date": str(frame.loc[usable, "race_date"].min().date()),
        "pool_fit_last_date": str(frame.loc[usable, "race_date"].max().date()),
        "components": {name: model.lineage for name, model in models.items()},
        "performance_claim": "Deployment candidate has no held-out performance estimate"}
    return WinCandidate(models, pool, lineage)


def save_candidate(candidate, receipt, *, name="win ensemble candidate", elapsed_seconds=0., evaluation=None):
    from src.run_store import save_run
    return save_run(name=name, model_type="win_stack_candidate", data_source="feature_store",
        data_rows=sum(m.lineage["training_runners"] for m in list(candidate.models.values())[:1]),
        n_features=len(set(n for m in candidate.models.values() for n in m.spec.features)),
        elapsed_seconds=elapsed_seconds, predictor=candidate,
        ensemble_weights=dict(zip(candidate.pool.components, candidate.pool.weights)),
        training_config={"signatures": {k: receipt[k] for k in ("fe_signature", "data_signature")},
            "dataset_manifest": receipt["dataset_manifest"], "split": {"mode": "fit_as_of", "holdout_races": 0,
                "training_last_date": candidate.lineage["as_of"]}, "modelling": candidate.lineage,
            "evaluation_reference": evaluation, "serving_qualified": False})
