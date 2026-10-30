from types import SimpleNamespace
import json
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.issued_evaluation import assess_issued
from src.deployment_validation import file_sha256


def study_and_results(tmp_path):
    folder = tmp_path/"records"/"capture"
    folder.mkdir(parents=True)
    frame = pd.DataFrame({"race_id":["r"]*3, "horse_key":["A","B","C"], "runner_id":["1","2","3"],
        "model_probability":[.5,.3,.2], "market_probability":[.4,.4,.2]})
    frame.to_parquet(folder/"forecasts.parquet", index=False)
    record = {"event_id":"1", "market_id":"m", "capture_id":"capture", "first_start":"2026-01-01T12:00:00Z",
              "published_at":"2026-01-01T11:55:00Z"}
    study = SimpleNamespace(path=tmp_path, selected_records=lambda: iter([record]))
    results = pd.DataFrame({"race_id":["r"]*2, "horse_name":["A","B"], "race_date":["2026-01-01"]*2,
        "finish_position":[1,2], "won":[1,0], "num_runners":[2,2]})
    return study, results


def withdrawal(study):
    receipt = study.path/"source.json"
    receipt.write_text(json.dumps({"event_id":"1", "observed_at":"2026-01-01T13:00:00Z",
        "payload":{"markets":[{"id":"m","runners":[{"id":"3","name":"C","withdrawn":True}]}]}}))
    return pd.DataFrame([{"event_id":"1", "horse_key":"C", "confirmed_withdrawn":True,
        "runner_id":"3", "market_id":"m", "source":"source.json", "source_sha256":file_sha256(receipt),
        "observed_at":"2026-01-01T13:00:00Z"}])


def test_as_issued_retains_withdrawn_forecast_mass(tmp_path):
    study, results = study_and_results(tmp_path)
    report = assess_issued(study, results, withdrawal(study))
    assert report["coverage"]["scored_fields"] == 1
    assert report["coverage"]["confirmed_withdrawn_runners"] == 1
    assert report["metrics"]["model"]["race_log_loss"] == pytest.approx(-np.log(.5))
    assert report["metrics"]["model"]["brier"] == pytest.approx((.5**2+.3**2+.2**2)/3)
    assert not report["renormalised_after_withdrawals"]


def test_absence_alone_is_not_a_withdrawal(tmp_path):
    study, results = study_and_results(tmp_path)
    report = assess_issued(study, results)
    assert "metrics" not in report
    assert report["coverage"]["absent_runner_without_confirmed_withdrawal"] == 1


@pytest.mark.parametrize("problem", ["unconfirmed", "preissue", "contradictory"])
def test_uncertain_withdrawals_cannot_create_certain_losers(tmp_path, problem):
    study, results = study_and_results(tmp_path)
    evidence = withdrawal(study)
    if problem == "unconfirmed": evidence.loc[0,"confirmed_withdrawn"] = False
    elif problem == "preissue": evidence.loc[0,"observed_at"] = "2025-12-31T10:00:00Z"
    else: evidence.loc[0,"horse_key"] = "B"
    with pytest.raises(ValueError):
        assess_issued(study, results, evidence)


def test_tampered_withdrawal_receipt_cannot_supply_labels(tmp_path):
    study, results = study_and_results(tmp_path)
    evidence = withdrawal(study)
    (study.path/"source.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        assess_issued(study, results, evidence)


def test_extreme_winner_is_reported_without_changing_its_forecast(tmp_path):
    study,results = study_and_results(tmp_path)
    path = study.path/"records"/"capture"/"forecasts.parquet"
    frame = pd.read_parquet(path)
    frame["model_probability"] = [.0001,.6999,.3]
    frame.to_parquet(path,index=False)
    report = assess_issued(study,results,withdrawal(study))
    assert report["tail_risk"]["model"]["one_tenth_percent"]["observed_wins"] == 1
    assert report["metrics"]["model"]["race_log_loss"] == pytest.approx(-np.log(.0001))
    assert report["tail_risk"]["model"]["maximum_winner_log_loss"] == pytest.approx(-np.log(.0001))


def test_hourly_result_snapshots_read_only_issued_races(tmp_path):
    from src.forward_assessment import load_results_readonly
    path = tmp_path/"results.db"
    labels = pd.DataFrame({"race_id":["issued","irrelevant"],"horse_name":["A","B"],
        "race_date":["2026-01-01"]*2,"finish_position":[1,1],"won":[1,1],"num_runners":[1,1]})
    with sqlite3.connect(path) as connection:
        labels.to_sql("results",connection,index=False)
    assert load_results_readonly(path,race_ids=["issued"]).race_id.tolist() == ["issued"]
    empty = load_results_readonly(path,race_ids=[])
    assert empty.empty and "finish_position" in empty
