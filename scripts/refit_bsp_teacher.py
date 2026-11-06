"""Install audited missing BSP observations and refit only the affected win head.

The input features and hard place labels stay identical. Reusing the place
head requires the same dates, rows, feature order and exact scaler state.
Normal future training reads the expanded default cache without a flag.
"""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.canonical_training_cache import CheckpointPredictor
from src.deployment_validation import file_sha256 as sha, training_recipe, load_validation_report
from src.data_processor import process_data
from src.feature_store import FeatureStore
from src.featured_dataset import fe_signature, data_signature
from src.model import RacePredictor, _IdentityScaler
from src.run_store import get_production_run_id, load_run_meta, promote, RUNS_DIR
from src.training import TrainingConfig

OUTPUT = ROOT / "data/audit/027_market_teacher_refresh"
CORRECTIONS = ROOT / "data/audit/canonical_repairs_026"


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def reuse_place(target, parent, origin, *, refresh_refinement=False):
    if target.feature_cols != parent.feature_cols or target.frameworks != parent.frameworks:
        raise ValueError("Place reuse requires identical inputs and frameworks")
    for key in ("as_of", "training_runners", "training_races", "training_first_date", "training_last_date", "calibration"):
        left,right = target.eval_split_info.get(key),parent.eval_split_info.get(key)
        if key == "calibration" and refresh_refinement:
            left = {k:v for k,v in (left or {}).items() if k != "place_refinement"}
            right = {k:v for k,v in (right or {}).items() if k != "place_refinement"}
        if left != right:
            raise ValueError(f"Place reuse requires identical training/calibration scope: {key}")
    if getattr(parent,"place_refinement",None) is not None and not refresh_refinement:
        raise ValueError("A teacher-dependent place refinement requires separate recalibration")
    if type(target.scaler) is not type(parent.scaler):
        raise ValueError("Place reuse requires identical preprocessing")
    if isinstance(target.scaler, _IdentityScaler):
        # Native tree inputs use a stateless float32 conversion. Its class
        # identity and absence of fitted state fully define the transform.
        if vars(target.scaler) or vars(parent.scaler):
            raise ValueError("Unexpected state in the identity input transform")
    else:
        if target.scaler.get_params() != parent.scaler.get_params():
            raise ValueError("Place reuse requires identical preprocessing")
        for key in ("mean_", "var_", "scale_", "n_samples_seen_", "n_features_in_"):
            np.testing.assert_array_equal(getattr(target.scaler, key), getattr(parent.scaler, key), err_msg="Place preprocessing changed")
    if parent.place_model is None:
        raise ValueError("Parent has no fitted place head")
    for key in ("place_model", "place_platt_a", "place_platt_b", "place_iso_cal"):
        setattr(target, key, getattr(parent, key))
    target.eval_split_info["reused_place_component"] = origin
    target.eval_split_info["refit_rounds"]["place"] = parent.eval_split_info["refit_rounds"]["place"]


class TeacherRefreshPredictor(CheckpointPredictor):
    def __init__(self, *args, parent_path, parent_origin, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent_path, self._parent_origin = Path(parent_path), parent_origin

    def fit_as_of(self, *args, **kwargs):
        kwargs["win_only"] = True
        result = super().fit_as_of(*args, **kwargs)
        parent = RacePredictor()
        parent.load(str(self._parent_path))
        reuse_place(self, parent, self._parent_origin,refresh_refinement=True)
        if getattr(parent,"place_refinement",None) is not None:
            # Reuse the final place estimator, not a calibration layer whose
            # earlier win-probability floors depend on the changed teacher.
            import config
            import pandas as pd
            from src.model import prepare_multi_target_data
            from src.probability_calibration import fit_recent_place_refinement
            frame = args[0]
            cutoff = args[1] if len(args)>1 else kwargs["as_of"]
            allowed = frame.loc[pd.to_datetime(frame.race_date).le(cutoff)].copy()
            data = prepare_multi_target_data(allowed,dataset_start=kwargs.get("dataset_start"),training_only=True)
            groups = data["groups_train"]
            starts = np.r_[0,np.cumsum(groups)[:-1]]
            teacher = data["win_teacher_train"]
            soft = (np.where(teacher>0,config.WIN_SOFT_LABEL_BLEND*teacher+(1-config.WIN_SOFT_LABEL_BLEND)*data["y_train_won"],data["y_train_won"])
                    if config.WIN_SOFT_LABEL_DISTILL else None)
            rows = data["train_df"]
            self.place_refinement,detail = fit_recent_place_refinement(self,data["X_train"],groups,
                data["y_train_won"],data["y_train_placed"],data["fp_train"],
                pd.DatetimeIndex(pd.to_datetime(rows.race_date).to_numpy()[starts]),soft_y=soft,model_params=kwargs.get("params"),
                race_types=rows.race_type.to_numpy() if "race_type" in rows else None,
                handicap=rows.handicap.to_numpy() if "handicap" in rows else None)
            self.eval_split_info["calibration"]["place_refinement"] = detail
        self.eval_split_info["win_only"] = False
        return dict(self.eval_split_info)


def run():
    from src import training
    state = json.loads((CORRECTIONS / "deployment_status.json").read_text())
    if state["stage"] != "installed":
        raise ValueError("Finish and verify the complete026 correction refit first")
    parent_id = state["run_id"]
    parent_path = Path(RUNS_DIR) / parent_id / "model.joblib"
    parent_meta = load_run_meta(parent_id)
    signatures = parent_meta["training_config"]["signatures"]
    store = FeatureStore().meta()
    if signatures["fe_signature"] != fe_signature() or any(signatures[k] != store[k] for k in ("fe_signature", "data_signature")):
        raise ValueError("Teacher-only refresh requires the same corrected input generation")
    processed = process_data()
    if data_signature(processed, cutoff=processed.race_date.max()) != signatures["data_signature"]:
        raise ValueError("New historical information requires a complete matching refit, not place-head reuse")
    audit = json.loads((OUTPUT / "candidate_audit.json").read_text())
    candidate = OUTPUT / "candidate_bsp.parquet"
    canonical = ROOT / "data/processed/betfair_bsp.parquet"
    if sha(candidate) != audit["candidate_sha256"] or audit["complete_candidate_training_fields"] <= 63977:
        raise ValueError("The expanded teacher candidate failed its data audit")
    if (audit["eligible_training_fields"] != parent_meta["training_config"]["split"]["training_races"]
            or audit["complete_candidate_training_fields"] > audit["eligible_training_fields"]):
        raise ValueError("Teacher audit does not cover the parent training population")
    cfg = TrainingConfig.deployment(CORRECTIONS / "validation.json", store["last_date"],
        name="027 expanded BSP teacher: same corrected features", refresh_data=False, threads=2)
    load_validation_report(CORRECTIONS / "validation.json", training_recipe(cfg))
    manifest_path = OUTPUT / "manifest.json"
    sources = ("scripts/refit_bsp_teacher.py", "scripts/canonical_training_cache.py", "src/betfair.py",
               "data/audit/027_market_teacher_refresh/prepare_candidate.py",
               "data/audit/027_market_teacher_refresh/stage_public_prices.py")
    if not manifest_path.exists():
        if get_production_run_id() != parent_id or sha(canonical) != audit["original_sha256"]:
            raise ValueError("The parent or canonical teacher changed before installation")
        manifest = {"parent_run": parent_id, "parent_model_sha256": sha(parent_path),
            "recipe": training_recipe(cfg), "signatures": signatures,
            "source_sha256": {name: sha(ROOT / name) for name in sources},
            "old_teacher_sha256": audit["original_sha256"], "new_teacher_sha256": audit["candidate_sha256"],
            "data_audit_sha256": sha(OUTPUT / "candidate_audit.json"),
            "policy": "Audited source-data extension under the same BSP-preferred training recipe; no new performance estimate or reserved-period scoring"}
        write(manifest_path, manifest)
    manifest = json.loads(manifest_path.read_text())
    if training_recipe(cfg) != manifest["recipe"] or sha(parent_path) != manifest["parent_model_sha256"]:
        raise ValueError("The recorded teacher-refresh recipe or parent changed")
    if sha(candidate) != manifest["new_teacher_sha256"] or sha(OUTPUT / "candidate_audit.json") != manifest["data_audit_sha256"]:
        raise ValueError("The recorded teacher candidate or data audit changed")
    if any(sha(ROOT / name) != digest for name, digest in manifest["source_sha256"].items()):
        raise ValueError("Teacher-refresh execution source changed")
    backup = OUTPUT / "previous_bsp.parquet"
    if sha(canonical) == manifest["old_teacher_sha256"]:
        if not backup.exists():
            shutil.copy2(canonical, backup)
        if sha(backup) != manifest["old_teacher_sha256"]:
            raise ValueError("The previous teacher cache was not preserved correctly")
        temporary = canonical.with_name(f".bsp-refresh-{uuid4().hex}.parquet")
        shutil.copy2(candidate, temporary)
        temporary.replace(canonical)
    if sha(canonical) != manifest["new_teacher_sha256"]:
        raise ValueError("The installed teacher cache differs from the audited candidate")
    progress = OUTPUT / "refit_status.json"
    saved = json.loads(progress.read_text()) if progress.exists() else {}
    run_id = saved.get("run_id")
    if not run_id:
        if get_production_run_id() != parent_id:
            raise ValueError("Installed parent changed before the win refit")
        ordinary = training.RacePredictor
        ordinary_processing = training.process_data
        training.RacePredictor = partial(TeacherRefreshPredictor, parent_path=parent_path,
            parent_origin={"run_id": parent_id, "model_sha256": manifest["parent_model_sha256"],
                "scope": "Same cutoff, complete hard labels, feature order and exact scaler"},
            checkpoint_directory=OUTPUT / "components", checkpoint_manifest=sha(manifest_path))
        # Reuse the exact normal processed frame already checked above;
        # repeat parsing/backfilling adds time without changing any inputs.
        training.process_data = lambda: processed
        write(progress, {"stage": "win_refit", "parent_run": parent_id})
        try:
            run_id = training.train_run(cfg)
        finally:
            training.RacePredictor = ordinary
            training.process_data = ordinary_processing
        write(progress, {"stage": "saved", "run_id": run_id})
    if (training_recipe(cfg) != manifest["recipe"] or sha(canonical) != manifest["new_teacher_sha256"]
            or any(sha(ROOT / name) != digest for name, digest in manifest["source_sha256"].items())):
        raise ValueError("Frozen teacher-refresh inputs changed during fitting")
    if get_production_run_id() not in (parent_id, run_id):
        raise ValueError("Another model was installed during the teacher refresh")
    if load_run_meta(run_id)["training_config"]["signatures"] != manifest["signatures"]:
        raise ValueError("Teacher refresh changed the recorded feature inputs")
    if not promote(run_id):
        raise ValueError("Expanded-teacher model could not be installed")
    write(progress, {"stage": "installed", "run_id": run_id, "parent_run": parent_id,
        "model_sha256": sha(Path(RUNS_DIR) / run_id / "model.joblib"),
        "scope": manifest["policy"]})
    print(progress.read_text(), flush=True)


if __name__ == "__main__":
    run()
