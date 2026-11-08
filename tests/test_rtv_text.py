from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ingestion import rtv_observations as collection
from src import rtv_text


def _field(race, names, comments, verdict, won=None, race_date="2021-01-05"):
    return pd.DataFrame({"race_id": race, "horse_id": range(len(names)), "race_date": race_date, "horse_name": names,
                         "tf_comment": comments, "tf_verdict": verdict,
                         "won": won if won is not None else [1]+[0]*(len(names)-1)})


def test_verdict_order_and_sentences_use_names_without_country_suffix():
    frame = rtv_text.prepare(_field("r1", ["Lady Phillips (IRE)", "Gone By", "Orionis"], ["a", "b", "c"],
        "GONE BY can win this. Lady Phillips is the danger. Others are outsiders."))
    assert frame.verdict_rank.tolist() == [2, 1, 0]
    assert frame.verdict_text.tolist() == ["Lady Phillips is the danger.", "GONE BY can win this.", ""]


def test_text_observations_key_rtv_text_to_sl_starters():
    races = pd.DataFrame({"race_id": [10], "race_date": ["2025-01-03"], "card_analyst_verdict_text": ["A wins."]})
    runners = pd.DataFrame({"race_id": [10, 10], "runner_id": [1, 2], "card_analyst_comment_text": ["nice", None]})
    links = pd.DataFrame({"rtv_race_id": [10, 10], "rtv_runner_id": [1, 2], "sl_race_id": ["900", "900"],
                          "sl_horse_id": [5, 6], "race_date": ["2025-01-03"]*2})
    out = rtv_text.build_text_observations(races, runners, links)
    assert out.race_id.tolist() == ["900", "900"] and out.horse_id.tolist() == [5, 6]
    assert out.race_date.eq("2025-01-03").all() and out.tf_verdict.eq("A wins.").all()
    assert out.tf_comment.tolist()[0] == "nice" and out.tf_comment.isna().tolist()[1]


def test_a_models_vocabulary_counts_exactly_what_the_fit_counted():
    frame = rtv_text.prepare(pd.concat([
        _field("r1", ["A", "B"], ["stays well and travels strongly", "lost his way"], "A wins."),
        _field("r2", ["C", "D"], ["travels strongly", None], None)], ignore_index=True))
    full, names = rtv_text.counts(frame.assign(tf_comment=frame.tf_comment.fillna("")))
    keep = [n for n in names if n.startswith("c:travels") or n.startswith("rank:")]
    fixed, fixed_names = rtv_text.counts(frame, vocabulary=np.asarray(keep))
    assert fixed_names == keep
    assert (fixed != full[:, [names.index(n) for n in keep]]).nnz == 0


def test_a_quarter_model_never_sees_races_after_its_purged_cutoff():
    rng = np.random.default_rng(0)
    words = ["travels well", "stays", "lost", "improver", "weak", "strong"]
    races = []
    for i, day in enumerate(pd.date_range("2020-10-01", "2021-03-31", freq="D")):
        comments = [" ".join(rng.choice(words, 2)) for _ in range(4)]
        races.append(_field(f"r{i:03d}", ["A", "B", "C", "D"], comments, None,
                            won=list(rng.permutation([1, 0, 0, 0])), race_date=day.strftime("%Y-%m-%d")))
    frame = rtv_text.prepare(pd.concat(races, ignore_index=True))
    monkey = rtv_text.MIN_DF
    rtv_text.MIN_DF = 1
    try:
        first = list(rtv_text.fit_models(frame, ["2021Q1"], purge_days=7))[0]
        later = pd.to_datetime(frame.race_date) > pd.Timestamp("2020-12-25")
        changed = frame.assign(won=np.where(later, frame.won[::-1].to_numpy(), frame.won))
        second = list(rtv_text.fit_models(changed, ["2021Q1"], purge_days=7))[0]
    finally:
        rtv_text.MIN_DF = monkey
    assert first.trained_through == "2020-12-25"
    np.testing.assert_array_equal(first.names, second.names)
    np.testing.assert_allclose(first.weights, second.weights)
    assert first.scale == second.scale


def test_history_fills_only_rows_without_a_snapshot_and_prob_is_a_race_softmax():
    frame = pd.DataFrame({"race_id": ["x", "x", "x", "y"], "horse_id": [1, 2, 3, 4],
                          "rtv_text_logit": [0.5, np.nan, np.nan, np.nan], "rtv_verdict_rank": [1., np.nan, np.nan, np.nan]})
    scores = pd.DataFrame({"race_id": ["x", "x", "y"], "horse_id": [1, 2, 4], "race_date": "2025-01-03",
                           "rtv_text_logit": [9., 0.5, 2.], "rtv_verdict_rank": [3., 0., 1.]})
    out = rtv_text.add_text_features(frame, scores=scores)
    assert out.rtv_text_logit.tolist()[:2] == [0.5, 0.5]       # the snapshot value wins
    assert np.isnan(out.rtv_text_logit.iloc[2])                  # no score anywhere
    assert out.rtv_text_prob.iloc[:2].tolist() == [0.5, 0.5]     # shared among scored runners only
    assert np.isnan(out.rtv_text_prob.iloc[2]) and out.rtv_text_prob.iloc[3] == 1.


def test_snapshot_scores_rank_only_runners_still_in_the_race(monkeypatch):
    key = "2025-08-14/kempton-park/1230"
    model = SimpleNamespace(logits=lambda part: part.verdict_rank.to_numpy(dtype=float)*10)
    monkeypatch.setattr(collection, "text_model", lambda quarter: model)
    frame = pd.DataFrame({"race_key": [key]*3, "fetched_at": ["2025-08-14T09:00:00Z"]*3,
        "card_horse_name": ["Alpha", "Beta", "Gamma"], "card_withdrawn": [True, False, False],
        "card_analyst_comment": ["<b>Fine</b> type", "ok", "ok"],
        "card_analyst_verdict": ["ALPHA was the pick. Gamma beats Beta."]*3})
    out = collection._snapshot_text(frame)
    assert np.isnan(out.rtv_text_logit.iloc[0])
    assert out.rtv_verdict_rank.iloc[1:].tolist() == [2., 1.]
    assert out.rtv_text_logit.iloc[1:].tolist() == [20., 10.]


@pytest.mark.parametrize("columns", [{}, {"card_analyst_comment": [None]}])
def test_snapshot_without_a_model_or_text_keeps_the_numeric_card(monkeypatch, columns):
    def missing(quarter):
        raise FileNotFoundError("no history")
    monkeypatch.setattr(collection, "text_model", missing)
    frame = pd.DataFrame({"race_key": ["2025-08-14/kempton-park/1230"], "fetched_at": ["2025-08-14T09:00:00Z"],
                          "card_horse_name": ["Alpha"], **columns})
    assert collection._snapshot_text(frame).isna().all().all()


def test_a_runner_without_text_is_missing_not_a_shared_constant():
    frame = rtv_text.prepare(pd.concat([
        _field("r1", ["A", "B", "C"], ["stays well", None, ""], "A wins."),
        _field("r2", ["D", "E"], ["travels", None], None)], ignore_index=True))
    names = np.asarray(["c:stays", "c:travels", "v:wins", *(f"rank:{r}" for r in range(rtv_text.RANKS))])
    model = rtv_text.TextModel("2021Q1", names, np.ones(len(names), np.float32), np.arange(len(names), dtype=float),
                               "2020-12-25", 10, 2.0)
    assert np.isfinite(model.logits(frame)).all()                # the model itself scores every row
    logits = rtv_text.text_logits(model, frame)
    assert np.isfinite(logits[[0, 3]]).all() and np.isnan(logits[[1, 2, 4]]).all()
    # Verdict order means 'not named' only in a race that has a verdict.
    np.testing.assert_array_equal(rtv_text.verdict_rank_feature(frame), [1., 0., 0., np.nan, np.nan])


def test_the_scale_recovers_how_over_confident_scores_are():
    rng = np.random.default_rng(1)
    races, size = 4000, 8
    score = rng.normal(0, 2, races*size)
    race_ids = np.repeat(np.arange(races), size)
    truth = rtv_text.race_probs(0.5*score, race_ids)
    won = np.zeros(races*size)
    for race in range(races):
        won[race*size + rng.choice(size, p=truth[race*size:(race+1)*size])] = 1
    assert abs(rtv_text.fit_scale(score, won, race_ids) - 0.5) < 0.05


def test_calibration_rows_are_the_latest_training_races_after_a_purge():
    dates = pd.date_range("2020-01-01", "2021-06-30", freq="D").to_numpy()
    train = np.arange(len(dates))
    fitting, held = rtv_text.calibration_split(dates, train, purge_days=7)
    first_held = pd.Timestamp(dates[held].min())
    assert first_held == pd.Timestamp("2021-06-30") - pd.Timedelta(days=rtv_text.CALIBRATION_DAYS)
    assert pd.Timestamp(dates[fitting].max()) == first_held - pd.Timedelta(days=7)
    short = train[:100]                     # under a year: its latest quarter of rows
    fitting, held = rtv_text.calibration_split(dates, short, purge_days=7)
    assert len(held) == 25 and dates[fitting].max() < dates[held].min()


def test_models_of_an_older_format_are_not_loaded(tmp_path):
    model = rtv_text.TextModel("2021Q1", np.asarray(["rank:0"]), np.ones(1, np.float32), np.zeros(1), "2020-12-25", 1, 0.7)
    model.save(tmp_path)
    assert rtv_text.load_model("2021Q1", tmp_path).scale == 0.7
    np.savez_compressed(tmp_path/"2021Q2.npz", names=np.asarray(["rank:0"]), idf=np.ones(1), weights=np.zeros(1),
                        trained_through="2021-03-25", rows=1)
    assert rtv_text.load_model("2021Q2", tmp_path) is None
    assert rtv_text.models_dir().name == f"format_{rtv_text.FORMAT}"


def test_stored_scores_from_an_older_format_force_a_full_rescore(tmp_path):
    def stored(name, **columns):
        path = tmp_path/name
        pd.DataFrame({"race_id": ["1"]*len(next(iter(columns.values()))), **columns}).to_parquet(path)
        return path
    assert collection._older_format(stored("old.parquet", rtv_text_logit=[0.1]))
    assert collection._older_format(stored("mixed.parquet", rtv_text_format=[rtv_text.FORMAT, 1]))
    assert not collection._older_format(stored("current.parquet", rtv_text_format=[rtv_text.FORMAT]))
    assert not collection._older_format(tmp_path/"absent.parquet")


def test_quarter_models_train_only_on_races_whose_whole_field_has_text(monkeypatch, tmp_path):
    frame = rtv_text.prepare(pd.concat([
        _field("r1", ["A", "B"], ["stays", "travels"], None, race_date="2020-10-01"),
        _field("r2", ["C", "D"], ["stays", None], None, race_date="2020-10-02"),
        _field("r3", ["E", "F"], ["stays", "travels"], None, won=[0, 0], race_date="2020-10-03")], ignore_index=True))
    seen = {}

    def fit(rows, labels, **kwargs):
        seen["races"] = sorted(rows.race_id.unique())
        return iter(())
    monkeypatch.setattr(collection, "_text_rows", lambda *args, **kwargs: frame)
    monkeypatch.setattr(rtv_text, "fit_models", fit)
    monkeypatch.setattr(rtv_text.config, "RTV_TEXT_MODELS", tmp_path, raising=False)
    collection.ensure_text_models("2021-01-05")
    assert seen["races"] == ["r1"]
