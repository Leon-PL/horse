"""An explicit text expert; fitting is never triggered from prediction."""
from dataclasses import dataclass

import numpy as np

from src import rtv_text
from src.modelling.data import ordered_frame
from src.race_objectives import race_bounds, race_softmax


@dataclass
class TextExpert:
    id: str = "timeform-text"
    penalty: float = 3e-5
    min_df: int = 20
    model: object = None
    lineage: dict = None

    def fit(self, frame):
        ordered, _ = ordered_frame(frame)
        if not {"tf_comment", "tf_verdict"}.issubset(ordered):
            raise ValueError("Text fitting requires explicitly supplied comments and verdicts")
        if not np.isfinite(self.penalty) or self.penalty <= 0 or self.min_df < 5:
            raise ValueError("Invalid text fitting settings")
        prepared = rtv_text.prepare(ordered)
        x, names = rtv_text.counts(prepared)
        keep = np.asarray((x > 0).sum(axis=0)).ravel() >= self.min_df
        if not keep.any():
            raise ValueError("Insufficient text support")
        x = x[:, keep]
        names = np.asarray(names)[keep]
        idf = rtv_text.idf(x)
        weights = rtv_text.fit_softmax(rtv_text.tfidf(x, idf), prepared.won.to_numpy(float), prepared.race_id, self.penalty)
        last = str(ordered.race_date.max().date())
        self.model = rtv_text.TextModel("explicit", names, idf, weights, last, len(ordered))
        self.lineage = {"training_last_date": last, "training_runners": len(ordered),
                        "terms": len(names), "penalty": self.penalty, "min_df": self.min_df}
        from src.modelling.evaluation import frame_signature
        self.lineage["input_signature"] = frame_signature(ordered,
            ["race_id", "horse_id", "race_date", "won", "tf_comment", "tf_verdict"])
        return self

    def predict(self, frame):
        if self.model is None:
            raise ValueError("Text expert is unavailable; fit it explicitly before prediction")
        ordered, positions = ordered_frame(frame, labelled=False)
        if not {"tf_comment", "tf_verdict"}.issubset(ordered):
            return np.full(len(frame), np.nan)
        prepared = rtv_text.prepare(ordered)
        present = prepared[["tf_comment", "tf_verdict"]].fillna("").astype(str).apply(lambda s: s.str.strip().ne("")).any(axis=1)
        covered = present.groupby(prepared.race_id).transform("any").to_numpy()
        p = race_softmax(self.model.logits(prepared), race_bounds(prepared.race_id))
        p[~covered] = np.nan
        out = np.empty(len(frame))
        out[positions] = p
        return out
