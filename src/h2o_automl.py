from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.model import prepare_multi_target_data

logger = logging.getLogger(__name__)


def _java_exe_from_home(home: str | Path | None) -> str | None:
    """Resolve a java executable from a JAVA_HOME-like directory."""
    if not home:
        return None
    p = Path(home)
    exe = p / "bin" / ("java.exe" if os.name == "nt" else "java")
    return str(exe) if exe.exists() else None


def _iter_candidate_java_homes() -> list[str]:
    """Collect likely Java install directories on Windows/Linux/macOS."""
    homes: list[str] = []

    # 1) Existing env config
    if os.environ.get("JAVA_HOME"):
        homes.append(os.environ["JAVA_HOME"])

    # 2) java on PATH
    java_on_path = shutil.which("java")
    if java_on_path:
        try:
            java_bin = Path(java_on_path).resolve().parent
            homes.append(str(java_bin.parent))
        except Exception:
            pass

    # 3) Common Windows install locations
    if os.name == "nt":
        roots = [
            Path(r"C:\Program Files\Java"),
            Path(r"C:\Program Files\Eclipse Adoptium"),
            Path(r"C:\Program Files\Microsoft"),
            Path(r"C:\Program Files\Amazon Corretto"),
        ]
        for root in roots:
            if not root.exists():
                continue
            for child in root.iterdir():
                if child.is_dir() and ("jdk" in child.name.lower() or "jre" in child.name.lower() or "temurin" in child.name.lower()):
                    homes.append(str(child))

        # 4) Downloads fallback for unpacked zip installs
        downloads = Path.home() / "Downloads"
        if downloads.exists():
            for child in downloads.iterdir():
                if not child.is_dir():
                    continue
                lname = child.name.lower()
                if "jdk" in lname or "jre" in lname or "openjdk" in lname or "java" in lname:
                    homes.append(str(child))
                    # Common zip layout: <folder>/jdk-XX/
                    for nested in child.iterdir():
                        if nested.is_dir() and ("jdk" in nested.name.lower() or "jre" in nested.name.lower()):
                            homes.append(str(nested))

    # De-dup while preserving order
    uniq: list[str] = []
    seen: set[str] = set()
    for h in homes:
        key = str(h)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq


def _ensure_java_runtime() -> tuple[bool, str]:
    """Ensure Java is discoverable for H2O by setting JAVA_HOME/PATH if needed."""
    java_exe = shutil.which("java")
    if java_exe:
        return True, java_exe

    for home in _iter_candidate_java_homes():
        exe = _java_exe_from_home(home)
        if not exe:
            continue
        try:
            subprocess.run([exe, "-version"], capture_output=True, timeout=8, check=False)
        except Exception:
            continue

        bin_dir = str(Path(exe).parent)
        os.environ["JAVA_HOME"] = str(Path(home))
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        logger.info("H2O Java runtime configured via JAVA_HOME=%s", home)
        return True, exe

    return False, (
        "Cannot find Java runtime. Set JAVA_HOME to your JDK folder "
        "(e.g. C:\\Program Files\\Java\\jdk-XX) and restart Streamlit."
    )


def h2o_is_available() -> tuple[bool, str]:
    """Check whether H2O and Java are available without failing app startup."""
    ok, detail = _ensure_java_runtime()
    if not ok:
        return False, detail
    try:
        import h2o  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, ""


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def _race_level_metrics(
    race_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | None]:
    if len(race_ids) == 0:
        return {"top1_accuracy": None, "ndcg_at_1": None}

    race_df = pd.DataFrame(
        {
            "race_id": race_ids,
            "y_true": y_true.astype(int),
            "y_prob": y_prob.astype(float),
        }
    )

    top_hits: list[float] = []
    ndcg_vals: list[float] = []
    for _, grp in race_df.groupby("race_id", sort=False):
        if grp.empty:
            continue
        top_row = grp.iloc[int(np.argmax(grp["y_prob"].values))]
        top_hits.append(float(top_row["y_true"]))
        if grp["y_true"].sum() > 0 and len(grp) > 1:
            try:
                ndcg_vals.append(
                    float(
                        ndcg_score(
                            [grp["y_true"].values.astype(float)],
                            [grp["y_prob"].values.astype(float)],
                            k=1,
                        )
                    )
                )
            except Exception:
                continue

    return {
        "top1_accuracy": _safe_float(np.mean(top_hits)) if top_hits else None,
        "ndcg_at_1": _safe_float(np.mean(ndcg_vals)) if ndcg_vals else None,
    }


def run_h2o_automl(
    featured_df: pd.DataFrame,
    *,
    target: str = "won",
    max_models: int = 20,
    max_runtime_secs: int | None = None,
    sort_metric: str = "AUC",
    seed: int | None = None,
    balance_classes: bool = True,
    exclude_algos: list[str] | None = None,
    nfolds: int = 5,
    use_blending: bool = False,
    blending_fraction: float = 0.15,
    include_stacked_ensembles: bool = True,
) -> dict[str, Any]:
    """Run H2O AutoML using the project's leak-safe train/test split."""
    _java_ok, _java_detail = _ensure_java_runtime()
    if not _java_ok:
        raise RuntimeError(_java_detail)

    import h2o
    from h2o.automl import H2OAutoML

    if target not in {"won", "placed"}:
        raise ValueError("target must be one of {'won', 'placed'}")

    payload = prepare_multi_target_data(featured_df)
    feature_cols = list(payload["feature_cols"])
    y_train = payload["y_train_won"] if target == "won" else payload["y_train_placed"]
    y_test = payload["y_test_won"] if target == "won" else payload["y_test_placed"]

    train_frame = pd.DataFrame(payload["X_train"], columns=feature_cols)
    train_frame["target"] = y_train.astype(int)

    test_frame = pd.DataFrame(payload["X_test"], columns=feature_cols)
    test_frame["target"] = y_test.astype(int)

    max_mem = getattr(config, "H2O_MAX_MEM_SIZE", "4G")
    h2o.init(max_mem_size=max_mem)

    train_hf = h2o.H2OFrame(train_frame)
    test_hf = h2o.H2OFrame(test_frame)
    train_hf["target"] = train_hf["target"].asfactor()
    test_hf["target"] = test_hf["target"].asfactor()

    final_exclude = list(exclude_algos or [])
    if not include_stacked_ensembles and "StackedEnsemble" not in final_exclude:
        final_exclude.append("StackedEnsemble")

    _nfolds = int(max(0, nfolds))
    if use_blending and _nfolds != 0:
        # Blending mode is designed for nfolds=0 in AutoML.
        _nfolds = 0

    automl = H2OAutoML(
        max_models=int(max_models),
        max_runtime_secs=(None if max_runtime_secs in (None, 0) else int(max_runtime_secs)),
        sort_metric=str(sort_metric),
        seed=int(seed if seed is not None else getattr(config, "RANDOM_SEED", 42)),
        balance_classes=bool(balance_classes),
        exclude_algos=final_exclude,
        nfolds=_nfolds,
        verbosity="warn",
    )

    blending_rows = 0
    effective_train_rows = int(len(train_frame))
    if use_blending:
        n_total = len(train_frame)
        blend_rows = int(max(200, round(n_total * float(blending_fraction))))
        blend_rows = min(blend_rows, max(0, n_total - 200))
        if blend_rows < 50:
            raise ValueError(
                "Not enough training rows to create a blending holdout. "
                "Use more data, reduce blending fraction, or disable blending."
            )
        base_df = train_frame.iloc[:-blend_rows].copy()
        blend_df = train_frame.iloc[-blend_rows:].copy()

        train_hf = h2o.H2OFrame(base_df)
        blend_hf = h2o.H2OFrame(blend_df)
        train_hf["target"] = train_hf["target"].asfactor()
        blend_hf["target"] = blend_hf["target"].asfactor()

        effective_train_rows = int(len(base_df))
        blending_rows = int(len(blend_df))
        automl.train(
            x=feature_cols,
            y="target",
            training_frame=train_hf,
            blending_frame=blend_hf,
        )
    else:
        automl.train(x=feature_cols, y="target", training_frame=train_hf)

    if automl.leader is None:
        raise RuntimeError("H2O AutoML completed without a leader model.")

    preds_df = automl.leader.predict(test_hf).as_data_frame(use_multi_thread=True)
    if "p1" in preds_df.columns:
        y_prob = preds_df["p1"].astype(float).values
    elif "p0" in preds_df.columns:
        y_prob = 1.0 - preds_df["p0"].astype(float).values
    else:
        y_prob = pd.to_numeric(preds_df.iloc[:, -1], errors="coerce").fillna(0.0).values.astype(float)

    y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)
    y_true = y_test.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics: dict[str, float | None] = {
        "brier": _safe_float(brier_score_loss(y_true, y_prob)),
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "precision": _safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["log_loss"] = _safe_float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        metrics["log_loss"] = None

    try:
        if np.unique(y_true).size >= 2:
            metrics["roc_auc"] = _safe_float(roc_auc_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None

    race_metrics = _race_level_metrics(
        race_ids=payload["test_df"]["race_id"].values,
        y_true=y_true,
        y_prob=y_prob,
    )
    metrics.update(race_metrics)

    leaderboard = automl.leaderboard.as_data_frame(use_multi_thread=True)

    return {
        "automl": automl,
        "leader": automl.leader,
        "leader_model_id": str(automl.leader.model_id),
        "leaderboard": leaderboard,
        "metrics": metrics,
        "n_train_rows": int(len(train_frame)),
        "n_train_rows_effective": int(effective_train_rows),
        "n_blending_rows": int(blending_rows),
        "n_test_rows": int(len(test_frame)),
        "n_features": int(len(feature_cols)),
        "target": target,
        "settings": {
            "nfolds": int(_nfolds),
            "use_blending": bool(use_blending),
            "blending_fraction": float(blending_fraction),
            "include_stacked_ensembles": bool(include_stacked_ensembles),
            "exclude_algos": list(final_exclude),
        },
    }


def save_h2o_leader_model(leader, path: str) -> str:
    """Persist the best H2O model to disk and return the saved path."""
    import h2o

    saved = h2o.save_model(model=leader, path=path, force=True)
    return str(saved)
