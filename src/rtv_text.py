"""Timeform comment score: a race-wise text model refitted every quarter (052).

The Racing TV card carries Timeform's analyst comment on each runner and a
race verdict. A race-wise softmax regression on their word and word-pair
counts (TF-IDF) gives each runner a text score. Each calendar quarter has its
own model, fitted only on races run before the quarter less the purge, so a
stored score never sees its own race or anything later.

History reads the backfilled card text, which matched the evening-before
snapshot on 2026-10-30 and does not change after the day (ledger 052). Live
cards read the pre-race snapshot and are scored by their quarter's model, the
same model that scored that quarter's history.

Fitting and collection I/O live in ``src.ingestion.rtv_observations``; this module
holds the model and the feature-engineering join.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.feature_extraction.text import CountVectorizer

import config

START = "2020-09-01"                 # first Racing TV card text
FIRST_QUARTER = "2021Q1"             # the two quarters before it only train
MIN_DF = 20                          # training rows an n-gram must appear in
PENALTY = 3e-5                       # L2 per weight, chosen in 052 on 2024Q1-2025Q2
FIELDS = {"tf_comment": "c", "verdict_text": "v"}   # text column -> n-gram prefix
TOKENS = r"(?u)\b[\w/'-]+\b"
RANKS = 5                            # verdict order dummies: 0 (not named), 1, 2, 3, 4+
SNAPSHOT_FEATURES = ("rtv_text_logit", "rtv_verdict_rank")
FEATURES = SNAPSHOT_FEATURES + ("rtv_text_prob",)


def text_path():
    from src.ingestion.rtv_storage import materialised_path
    from src.source_dataset import source_path
    return source_path("rtv", "text", Path(config.RTV_FEATURE_TEXT) if hasattr(config, "RTV_FEATURE_TEXT") else materialised_path("text", Path(config.DATA_DIR)/"rtv_cache/feature_text.parquet"))


def scores_path():
    from src.ingestion.rtv_storage import materialised_path
    from src.source_dataset import source_path
    return source_path("rtv", "scores", Path(config.RTV_FEATURE_TEXT_SCORES) if hasattr(config, "RTV_FEATURE_TEXT_SCORES") else materialised_path("scores", Path(config.DATA_DIR)/"rtv_cache/feature_text_scores.parquet"))


def models_dir():
    return Path(getattr(config, "RTV_TEXT_MODELS", Path(config.DATA_DIR)/"rtv_cache/text_models"))


# ------------------------------------------------------------------ text --

def build_text_observations(races, runners, links):
    """Comment and race verdict for each exact SL/RTV starter identity."""
    columns = ["race_id", "horse_id", "race_date", "tf_comment", "tf_verdict"]
    if links.empty:
        return pd.DataFrame(columns=columns)
    runner_text = runners.reindex(columns=["race_id", "runner_id", "card_analyst_comment_text"])
    race_text = races.reindex(columns=["race_id", "card_analyst_verdict_text"])
    base = (links.merge(runner_text, left_on=["rtv_race_id", "rtv_runner_id"], right_on=["race_id", "runner_id"],
                        validate="one_to_one").drop(columns=["race_id", "runner_id"])
            .merge(race_text, left_on="rtv_race_id", right_on="race_id", validate="many_to_one"))
    output = pd.DataFrame({"race_id": base.sl_race_id.astype(str), "horse_id": pd.to_numeric(base.sl_horse_id),
        "race_date": pd.to_datetime(base.race_date).dt.strftime("%Y-%m-%d"),
        "tf_comment": base.card_analyst_comment_text, "tf_verdict": base.card_analyst_verdict_text})
    has_text = output.tf_comment.notna() | output.tf_verdict.notna()
    return output.loc[has_text, columns].reset_index(drop=True)


def plain_names(names) -> pd.Series:
    """Horse names without the country suffix, as the verdict writes them."""
    names = pd.Series(names, dtype=object).fillna("").astype(str)
    return names.str.replace(r"\s*\([A-Z]{2,3}\)\s*$", "", regex=True).str.strip()


def verdict_sentences(verdict, name) -> str:
    """The verdict sentences that name the horse."""
    if not isinstance(verdict, str) or not name:
        return ""
    pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.I)
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", verdict) if pattern.search(s))


def verdict_ranks(verdicts, names) -> np.ndarray:
    """Order in which one race's verdict first names each runner (0 = not named)."""
    positions = []
    for verdict, name in zip(verdicts, names):
        match = re.search(r"\b" + re.escape(name) + r"\b", verdict, re.I) if isinstance(verdict, str) and name else None
        positions.append(match.start() if match else np.nan)
    return pd.Series(positions).rank(method="first").fillna(0).astype(int).to_numpy()


def prepare(frame, race="race_id"):
    """Add verdict order and the verdict sentences naming each runner.

    ``frame`` holds whole fields (``race``, ``horse_name``, ``tf_comment``,
    ``tf_verdict``): the order counts only the runners in the frame.
    """
    output = frame.copy()
    output["plain_name"] = plain_names(output.horse_name).to_numpy()
    verdicts, names = output.tf_verdict.to_numpy(), output.plain_name.to_numpy()
    ranks = np.zeros(len(output), dtype=int)
    for rows in output.groupby(race, sort=False).indices.values():
        ranks[rows] = verdict_ranks(verdicts[rows], names[rows])
    output["verdict_rank"] = ranks
    output["verdict_text"] = [verdict_sentences(v, n) for v, n in zip(verdicts, names)]
    return output


# ----------------------------------------------------------------- model --

def _vectorizer(**kwargs):
    return CountVectorizer(ngram_range=(1, 2), token_pattern=TOKENS, dtype=np.float32, **kwargs)


def counts(frame, vocabulary=None, fields=None):
    """Sparse n-gram and verdict-order counts, with their names.

    With no vocabulary, every n-gram in at least five rows is counted (a fold
    keeps those frequent in its own training rows). With a model's vocabulary,
    exactly its columns are counted, in its order.
    """
    blocks, names = [], []
    for column, prefix in (fields or FIELDS).items():
        docs = frame[column].fillna("").astype(str).str.casefold()
        if vocabulary is None:
            vectorizer = _vectorizer(min_df=5)
            try:
                blocks.append(vectorizer.fit_transform(docs))
                words = list(vectorizer.get_feature_names_out())
            except ValueError:  # no n-gram in five rows
                blocks.append(sparse.csr_matrix((len(frame), 0), dtype=np.float32))
                words = []
        else:
            words = [n[len(prefix)+1:] for n in vocabulary if n.startswith(prefix+":")]
            blocks.append(_vectorizer(vocabulary=words).transform(docs) if words
                          else sparse.csr_matrix((len(frame), 0), dtype=np.float32))
        names += [f"{prefix}:{w}" for w in words]
    rank = np.clip(frame.verdict_rank.to_numpy(dtype=int), 0, RANKS-1)
    blocks.append(sparse.csr_matrix((np.ones(len(frame), np.float32), (np.arange(len(frame)), rank)),
                                    shape=(len(frame), RANKS)))
    names += [f"rank:{r}" for r in range(RANKS)]
    x = sparse.hstack(blocks, format="csr")
    if vocabulary is not None:
        position = {n: i for i, n in enumerate(names)}
        names = list(vocabulary)
        x = x[:, [position[n] for n in names]]
    return x, names


def idf(x) -> np.ndarray:
    df = np.asarray((x > 0).sum(axis=0)).ravel()
    return (np.log((1 + x.shape[0]) / (1 + df)) + 1).astype(np.float32)


def tfidf(x, weights) -> sparse.csr_matrix:
    """Sublinear term frequency, IDF and unit row length."""
    x = x.tocsr(copy=True)
    x.data = 1 + np.log(x.data)
    x = x @ sparse.diags(weights)
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    return sparse.diags(1 / np.where(norms > 0, norms, 1)) @ x


def race_index(race_ids) -> tuple[np.ndarray, np.ndarray]:
    """Start and size of each race; a race's rows must be contiguous."""
    ids = pd.Series(np.asarray(race_ids)).reset_index(drop=True)
    starts = ids.ne(ids.shift()).to_numpy()
    starts[0] = True
    if ids[starts].duplicated().any():
        raise ValueError("Races must be contiguous")
    first = np.flatnonzero(starts)
    return first, np.diff(np.r_[first, len(ids)])


def _softmax_loss(w, x, y, starts, sizes, penalty):
    score = x @ w
    top = np.maximum.reduceat(score, starts)
    e = np.exp(score - np.repeat(top, sizes))
    z = np.add.reduceat(e, starts)
    p = e / np.repeat(z, sizes)
    loss = -(np.add.reduceat(y * score, starts) - np.log(z) - top).mean()
    grad = (x.T @ (p - y)) / len(starts)
    return loss + penalty * (w @ w), grad + 2 * penalty * w


def fit_softmax(x, won, race_ids, penalty, w0=None):
    """Race-wise multinomial logit: one winner per race (a dead heat shares it)."""
    starts, sizes = race_index(race_ids)
    y = won / np.repeat(np.add.reduceat(won, starts), sizes)
    result = minimize(_softmax_loss, np.zeros(x.shape[1]) if w0 is None else w0, jac=True, method="L-BFGS-B",
                      args=(x, y, starts, sizes, penalty), options={"maxiter": 400})
    return result.x


def race_probs(score, race_ids) -> np.ndarray:
    starts, sizes = race_index(race_ids)
    top = np.maximum.reduceat(score, starts)
    e = np.exp(score - np.repeat(top, sizes))
    return e / np.repeat(np.add.reduceat(e, starts), sizes)


def quarter_start(quarter) -> pd.Timestamp:
    return pd.Period(quarter, freq="Q").start_time


def quarters(dates) -> pd.Series:
    return pd.to_datetime(pd.Series(dates)).dt.to_period("Q").astype(str)


@dataclass(frozen=True)
class TextModel:
    quarter: str
    names: np.ndarray
    idf: np.ndarray
    weights: np.ndarray
    trained_through: str
    rows: int

    def logits(self, frame) -> np.ndarray:
        """Scores for prepared rows: comment, verdict sentences and verdict order."""
        x, _ = counts(frame, vocabulary=self.names)
        return tfidf(x, self.idf) @ self.weights

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/f"{self.quarter}.npz"
        temporary = path.with_name(path.stem+".tmp.npz")
        np.savez_compressed(temporary, names=self.names.astype(str), idf=self.idf, weights=self.weights,
                            trained_through=self.trained_through, rows=self.rows)
        temporary.replace(path)
        return path


@lru_cache(maxsize=8)
def _load(path, modified):
    with np.load(path, allow_pickle=False) as data:
        return TextModel(Path(path).stem, data["names"], data["idf"], data["weights"],
                         str(data["trained_through"]), int(data["rows"]))


def load_model(quarter, directory=None) -> TextModel | None:
    path = Path(directory or models_dir())/f"{quarter}.npz"
    return _load(str(path.resolve()), path.stat().st_mtime_ns) if path.exists() else None


def fit_models(frame, labels, *, purge_days, warm=None):
    """One model per quarter in ``labels``, each on the rows before it less the purge.

    ``frame`` is prepared, sorted by date and race, with ``won`` and only
    races that have a winner. Each fit starts from the previous quarter's
    weights, which only shortens the optimisation.
    """
    x, names = counts(frame)
    names = np.asarray(names)
    won = frame.won.to_numpy(np.float64)
    race_ids = frame.race_id.to_numpy()
    dates = pd.to_datetime(frame.race_date).to_numpy()
    for label in labels:
        cutoff = quarter_start(label) - pd.Timedelta(days=purge_days)
        train = np.flatnonzero(dates <= np.datetime64(cutoff))
        if not len(train):
            raise ValueError(f"No text history before {label}")
        columns = np.flatnonzero(np.asarray((x[train] > 0).sum(axis=0)).ravel() >= MIN_DF)
        sub = x[train][:, columns]
        weights_idf = idf(sub)
        w0 = None if warm is None else pd.Series(warm.weights, index=warm.names).reindex(names[columns]).fillna(0).to_numpy()
        w = fit_softmax(tfidf(sub, weights_idf), won[train], race_ids[train], PENALTY, w0=w0)
        warm = TextModel(label, names[columns], weights_idf, w, str(pd.Timestamp(dates[train].max()).date()), len(train))
        yield warm


# --------------------------------------------------- feature engineering --

def _scores_table(scores):
    scores = scores.assign(race_id=scores.race_id.astype(str), horse_id=pd.to_numeric(scores.horse_id, errors="coerce"))
    return scores.loc[~scores.duplicated(["race_id", "horse_id"], keep=False)]


@lru_cache(maxsize=2)
def _cached_scores(path, modified, size):
    from src.source_dataset import model_table
    return _scores_table(model_table("rtv", "scores", path))


def replaced_columns(frame, stale, output):
    """``frame`` without ``stale``, then ``output``, sharing blocks with ``frame``."""
    kept = frame.copy(deep=False)
    for column in stale:
        if column in kept.columns:
            del kept[column]
    return pd.concat([kept, output], axis=1, copy=False)


def add_text_features(frame, *, scores=None):
    """Fill rows without a pre-race snapshot from the stored history score.

    Snapshot values win: they are what a live forecast sees. ``rtv_text_prob``
    is the race-wise softmax over the runners in the frame that have a score.
    """
    current = frame.reindex(columns=list(SNAPSHOT_FEATURES)).to_numpy(dtype="float64", na_value=np.nan)
    missing = np.isnan(current).all(axis=1)
    path = scores_path()
    if scores is not None:
        scores = _scores_table(scores)
    elif path.exists():
        stat = path.stat()
        scores = _cached_scores(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if missing.any() and scores is not None and not scores.empty and {"race_id", "horse_id"}.issubset(frame):
        target = pd.DataFrame({"race_id": frame.race_id.astype(str).to_numpy(),
            "horse_id": pd.to_numeric(frame.horse_id, errors="coerce").to_numpy()})
        joined = target.merge(scores[["race_id", "horse_id", *SNAPSHOT_FEATURES]], on=["race_id", "horse_id"], how="left")
        current[missing] = joined.loc[missing, list(SNAPSHOT_FEATURES)].to_numpy(dtype="float64", na_value=np.nan)
    output = pd.DataFrame(current, index=frame.index, columns=list(SNAPSHOT_FEATURES))
    output["rtv_text_prob"] = np.nan
    if "race_id" in frame:
        logit = output.rtv_text_logit
        race = frame.race_id.astype(str)
        e = np.exp(logit - logit.groupby(race).transform("max"))
        output["rtv_text_prob"] = e / e.groupby(race).transform("sum")
    return replaced_columns(frame, FEATURES, output)
