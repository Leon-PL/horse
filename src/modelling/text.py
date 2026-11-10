"""Explicit calibrated sparse text experts; prediction never fits or reads sources."""
from dataclasses import dataclass, field
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.feature_extraction.text import CountVectorizer

from src import rtv_text
from src.betfair import market_prob_bsp_preferred
from src.modelling.data import ordered_frame
from src.race_objectives import training_objective, race_bounds, race_softmax


def normalise(value):
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).translate(str.maketrans({"’": "'", "‘": "'", "`": "'", "‑": "-", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", value).strip().casefold()


def prepare_roles(frame):
    """Give each named horse its own self/rival interpretation of the verdict."""
    comments, verdicts = np.empty(len(frame), object), np.empty(len(frame), object)
    ranks = np.zeros(len(frame), dtype=int)
    plain = rtv_text.plain_names(frame.horse_name).map(normalise).to_numpy()
    raw_comments, raw_verdicts = frame.tf_comment.to_numpy(), frame.tf_verdict.to_numpy()
    for rows in frame.groupby("race_id", sort=False).indices.values():
        names = set(plain[rows]) - {""}
        if len(names) != len(rows):
            raise ValueError("Text attribution requires unique nonempty runner names")
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in sorted(names, key=lambda n: (-len(n), n))) + r")(?!\w)")
        cache = {}
        for row in rows:
            name = plain[row]
            def replace(match):
                return "selfhorse" if match.group() == name else "rivalhorse"
            comments[row] = pattern.sub(replace, normalise(raw_comments[row]))
            verdict = normalise(raw_verdicts[row])
            if verdict not in cache:
                first = {}
                for match in pattern.finditer(verdict):
                    first.setdefault(match.group(), match.start())
                order = {horse: rank for rank, horse in enumerate(sorted(first, key=first.get), 1)}
                sentences = [(s, {m.group() for m in pattern.finditer(s)}) for s in re.split(r"(?<=[.!?])\s+", verdict)]
                cache[verdict] = order, sentences
            order, sentences = cache[verdict]
            ranks[row] = order.get(name, 0)
            verdicts[row] = " ".join(pattern.sub(replace, s) for s, mentioned in sentences if name in mentioned)
    return pd.DataFrame({"model_comment": comments, "model_verdict": verdicts, "model_rank": ranks}, index=frame.index)


@dataclass
class TextEncoder:
    min_df: int = 20
    ngram_max: int = 2
    roles: bool = False
    separate_fields: bool = False
    characters: bool = False
    blocks: list = field(default_factory=list)
    names: np.ndarray = None
    joint_idf: np.ndarray = None
    columns: np.ndarray = None

    def _counts(self, frame, *, fitting):
        if self.roles:
            values = frame if {"model_comment", "model_verdict", "model_rank"}.issubset(frame) else prepare_roles(frame)
            comment, verdict, rank = values.model_comment, values.model_verdict, values.model_rank.to_numpy()
        else:
            values = frame if {"verdict_text", "verdict_rank"}.issubset(frame) else rtv_text.prepare(frame)
            comment, verdict, rank = values.tf_comment, values.verdict_text, values.verdict_rank.to_numpy()
        fields = [("c", comment, False), ("v", verdict, False)]
        if self.characters:
            fields += [("cchar", comment, True), ("vchar", verdict, True)]
        blocks, names = [], []
        if fitting:
            self.blocks = []
        for i, (prefix, values, char) in enumerate(fields):
            docs = values.fillna("").astype(str).str.casefold()
            if fitting:
                settings = {"analyzer": "char_wb", "ngram_range": (3, 5), "max_features": 30000} if char else {
                    "ngram_range": (1, self.ngram_max), "token_pattern": rtv_text.TOKENS}
                vectorizer = CountVectorizer(min_df=self.min_df, dtype=np.float32, **settings)
                try:
                    x = vectorizer.fit_transform(docs)
                    terms = vectorizer.get_feature_names_out()
                except ValueError as error:
                    if not any(s in str(error) for s in ("empty vocabulary", "After pruning", "max_df corresponds")):
                        raise
                    vectorizer, x, terms = None, sparse.csr_matrix((len(frame), 0), dtype=np.float32), []
                weights = rtv_text.idf(x) if x.shape[1] else np.empty(0, dtype=np.float32)
                self.blocks.append((vectorizer, weights))
            else:
                vectorizer, weights = self.blocks[i]
                x = vectorizer.transform(docs) if vectorizer is not None else sparse.csr_matrix((len(frame), 0), dtype=np.float32)
                terms = vectorizer.get_feature_names_out() if vectorizer is not None else []
            blocks.append(rtv_text.tfidf(x, weights) if self.separate_fields and x.shape[1] else x)
            names.extend(f"{prefix}:{term}" for term in terms)
        rank = np.clip(np.asarray(rank, dtype=int), 0, rtv_text.RANKS - 1)
        blocks.append(sparse.csr_matrix((np.ones(len(frame), np.float32), (np.arange(len(frame)), rank)), shape=(len(frame), rtv_text.RANKS)))
        names.extend(f"rank:{r}" for r in range(rtv_text.RANKS))
        return sparse.hstack(blocks, format="csr"), np.asarray(names)

    def fit_transform(self, frame):
        x, names = self._counts(frame, fitting=True)
        if self.separate_fields:
            self.columns, self.names = np.arange(x.shape[1]), names
            return x
        self.columns = np.flatnonzero(np.asarray((x > 0).sum(axis=0)).ravel() >= self.min_df)
        x, self.names = x[:, self.columns], names[self.columns]
        self.joint_idf = rtv_text.idf(x)
        return rtv_text.tfidf(x, self.joint_idf)

    def transform(self, frame):
        if self.names is None:
            raise ValueError("Text encoder is not fitted")
        x, _ = self._counts(frame, fitting=False)
        return x if self.separate_fields else rtv_text.tfidf(x[:, self.columns], self.joint_idf)


@dataclass
class SparseTextModel:
    encoder: TextEncoder
    weights: np.ndarray
    scale: float
    trained_through: str
    rows: int
    optimisation: dict

    @property
    def names(self):
        return self.encoder.names

    def logits(self, frame):
        return self.scale * np.asarray(self.encoder.transform(frame) @ self.weights).ravel()


@dataclass
class TextExpert:
    id: str = "timeform-text"
    penalty: float = 3e-5
    min_df: int = 20
    ngram_max: int = 2
    roles: bool = False
    separate_fields: bool = False
    half_life_days: float = None
    stage_weights: tuple = (1.,)
    soft_label_blend: float = 0.
    characters: bool = False
    purge_days: int = 7
    max_iter: int = 400
    model: object = field(default=None, repr=False)
    lineage: dict = None

    def settings(self):
        return {k: getattr(self, k) for k in ("penalty", "min_df", "ngram_max", "roles", "separate_fields", "half_life_days",
            "stage_weights", "soft_label_blend", "characters", "purge_days", "max_iter")}

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(self.settings(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _fit(self, frame, encoding_provider=None):
        encoder = TextEncoder(min_df=self.min_df, ngram_max=self.ngram_max, roles=self.roles,
                              separate_fields=self.separate_fields, characters=self.characters)
        if encoding_provider is None:
            x = encoder.fit_transform(frame)
        else:
            x, encoder = encoding_provider(frame, encoder)
        if not sparse.issparse(x) or x.shape != (len(frame), len(encoder.names)):
            raise ValueError("Encoded text rows or feature names do not match")
        y = frame.won.to_numpy(dtype=float)
        if self.soft_label_blend:
            teacher = market_prob_bsp_preferred(frame)
            y = np.where(np.isfinite(teacher), (1 - self.soft_label_blend) * y + self.soft_label_blend * teacher, y)
        weights = np.ones(len(frame))
        groups = np.diff(race_bounds(frame.race_id))
        if self.half_life_days is not None:
            age = (frame.race_date.max() - frame.race_date).dt.days.to_numpy()
            weights = np.exp(-np.log(2) * age / self.half_life_days)
            weights /= weights[np.r_[0, np.cumsum(groups)[:-1]]].mean()
        objective = training_objective(groups, frame.finish_position.to_numpy(), y, weights, self.stage_weights)
        def loss(beta):
            value, gradient, _ = objective.loss_and_gradients(np.asarray(x @ beta).ravel())
            return value / len(groups) + self.penalty * (beta @ beta), np.asarray(x.T @ gradient).ravel() / len(groups) + 2 * self.penalty * beta
        result = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-6, "maxcor": 10})
        info = {"success": bool(result.success), "iterations": int(result.nit), "message": str(result.message),
                "gradient_max": float(np.abs(result.jac).max()), "terms": x.shape[1]}
        if (not np.isfinite(result.fun) or not np.isfinite(result.x).all() or not np.isfinite(result.jac).all()
                or not result.success and info["gradient_max"] > 1e-5):
            raise RuntimeError(f"Text optimisation did not converge: {info}")
        return SparseTextModel(encoder, result.x, 1., str(frame.race_date.max().date()), len(frame), info)

    def fit(self, frame, *, prepared=False, encoding_provider=None):
        self.model, self.lineage = None, None
        if (not np.isfinite(self.penalty) or self.penalty <= 0 or self.min_df < 5 or self.ngram_max not in (1, 2, 3)
                or not 0 <= self.soft_label_blend <= 1 or self.purge_days < 1 or self.max_iter < 1
                or not self.stage_weights or self.stage_weights[0] != 1 or any(not np.isfinite(w) or w < 0 for w in self.stage_weights)
                or self.half_life_days is not None and (not np.isfinite(self.half_life_days) or self.half_life_days <= 0)):
            raise ValueError("Invalid text fitting settings")
        ordered, _ = ordered_frame(frame)
        if not {"tf_comment", "tf_verdict"}.issubset(ordered):
            raise ValueError("Text fitting requires explicitly supplied comments and verdicts")
        rows = ordered if prepared else rtv_text.prepare(ordered.drop(columns=["model_comment", "model_verdict", "model_rank"], errors="ignore"))
        present = pd.Series(rtv_text.has_text(rows)).groupby(rows.race_id).transform("all").to_numpy()
        rows = rows.loc[present].reset_index(drop=True)
        if len(rows) < 100:
            raise ValueError("Insufficient complete text fields")
        fitting, calibration = rtv_text.calibration_split(rows.race_date.to_numpy(), np.arange(len(rows)), self.purge_days)
        if len(fitting) < 50 or len(calibration) < 20:
            raise ValueError("Insufficient earlier text history for calibration")
        temporary = self._fit(rows.iloc[fitting], encoding_provider)
        held = rows.iloc[calibration]
        scale = rtv_text.fit_scale(temporary.logits(held), held.won.to_numpy(float), held.race_id)
        model = self._fit(rows, encoding_provider)
        model.scale = scale
        self.model = model
        from src.modelling.evaluation import frame_signature
        self.lineage = {"training_last_date": model.trained_through, "training_runners": len(rows),
            "terms": len(model.names), "recipe": self.settings(), "recipe_signature": self.signature,
            "calibration_first_date": str(held.race_date.min().date()), "calibration_last_date": str(held.race_date.max().date()),
            "calibration_fit_boundary": str(rows.iloc[fitting].race_date.max().date()), "scale": scale,
            "optimisation": model.optimisation, "calibration_optimisation": temporary.optimisation,
            "input_signature": frame_signature(rows, ["race_id", "horse_id", "horse_name", "race_date", "won", "finish_position", "tf_comment", "tf_verdict"]
                + ([c for c in ("bsp", "odds") if c in rows] if self.soft_label_blend else []))}
        return self

    def predict(self, frame, *, prepared=False):
        if self.model is None:
            raise ValueError("Text expert is unavailable; fit it explicitly before prediction")
        ordered, positions = ordered_frame(frame, labelled=False)
        if not {"tf_comment", "tf_verdict"}.issubset(ordered):
            return np.full(len(frame), np.nan)
        rows = ordered if prepared else rtv_text.prepare(ordered.drop(columns=["model_comment", "model_verdict", "model_rank"], errors="ignore"))
        present = pd.Series(rtv_text.has_text(rows)).groupby(rows.race_id).transform("all").to_numpy()
        p = race_softmax(self.model.logits(rows), race_bounds(rows.race_id))
        p[~present] = np.nan
        out = np.empty(len(frame))
        out[positions] = p
        return out
