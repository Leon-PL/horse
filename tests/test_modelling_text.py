"""Text attribution, forward calibration, coverage and sparse feature contracts."""
import joblib
import numpy as np
import pandas as pd

from src import rtv_text
from src.modelling.text import TextEncoder, TextExpert, prepare_roles


def fields(days=120):
    rows = []
    for day in range(days):
        for horse in range(4):
            rows.append({"race_id": f"race{day:03}", "horse_id": horse, "horse_name": ["Alpha", "Beta", "Gamma", "Delta"][horse],
                "race_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=day), "num_runners": 4,
                "won": int(horse == day % 4), "finish_position": 1 + (horse - day) % 4,
                "tf_comment": "improving and stays well" if horse == day % 4 else "needs to improve",
                "tf_verdict": "Alpha is preferred to Beta. Gamma has place claims.", "odds": 5., "bsp": 4.})
    return pd.DataFrame(rows)


def test_classic_encoder_reproduces_existing_numeric_representation():
    frame = rtv_text.prepare(fields())
    old, names = rtv_text.counts(frame)
    keep = np.asarray((old > 0).sum(axis=0)).ravel() >= 5
    old = old[:, keep]
    expected = rtv_text.tfidf(old, rtv_text.idf(old))
    encoder = TextEncoder(min_df=5)
    actual = encoder.fit_transform(frame)
    np.testing.assert_array_equal(encoder.names, np.asarray(names)[keep])
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=1e-7)
    np.testing.assert_array_equal(encoder.transform(frame).toarray(), actual.toarray())


def test_roles_distinguish_horses_in_the_same_sentence_and_respect_long_names():
    frame = fields(1)
    role = prepare_roles(frame)
    assert role.model_verdict.iloc[0] == "selfhorse is preferred to rivalhorse."
    assert role.model_verdict.iloc[1] == "rivalhorse is preferred to selfhorse."
    assert role.model_verdict.iloc[3] == ""
    assert role.model_rank.tolist() == [1, 2, 3, 0]
    frame.loc[0, "horse_name"] = "Alpha Beta"
    frame["tf_verdict"] = "ALPHA BETA is preferred."
    assert prepare_roles(frame).model_rank.tolist() == [1, 0, 0, 0]


def test_unicode_quotes_and_case_have_consistent_name_attribution():
    frame = fields(1)
    frame.loc[0, "horse_name"] = "D'Artagnan (IRE)"
    frame["tf_verdict"] = "D’ARTAGNAN is preferred to BETA."
    result = prepare_roles(frame)
    assert result.model_rank.tolist() == [1, 2, 0, 0]
    assert "selfhorse is preferred to rivalhorse" in result.model_verdict.iloc[0]


def test_vocabulary_and_idf_do_not_learn_future_words():
    frame = rtv_text.prepare(fields())
    encoder = TextEncoder(min_df=5, ngram_max=3, separate_fields=True)
    encoder.fit_transform(frame.iloc[:240])
    names = encoder.names.copy()
    future = frame.iloc[240:].copy()
    future["tf_comment"] = "neverobservedtoken"
    encoder.transform(future)
    np.testing.assert_array_equal(names, encoder.names)
    assert not any("neverobservedtoken" in n for n in encoder.names)


def test_text_fit_calibrates_on_later_rows_and_reload_has_identical_forecasts(tmp_path):
    frame = fields()
    expert = TextExpert(min_df=5, roles=True, separate_fields=True).fit(frame)
    assert pd.Timestamp(expert.lineage["calibration_first_date"]) - pd.Timestamp(expert.lineage["calibration_fit_boundary"]) >= pd.Timedelta(days=7)
    inference = frame.drop(columns=["won", "finish_position", "bsp", "odds"])
    p = expert.predict(inference)
    np.testing.assert_allclose(pd.Series(p).groupby(frame.race_id).sum(), 1., atol=1e-12)
    shuffled = inference.sample(frac=1, random_state=2)
    np.testing.assert_allclose(expert.predict(shuffled), p[shuffled.index], atol=1e-12)
    path = tmp_path / "text.joblib"
    joblib.dump(expert, path)
    np.testing.assert_array_equal(joblib.load(path).predict(inference), p)
    missing = inference.copy()
    missing.loc[0, ["tf_comment", "tf_verdict"]] = None
    assert np.isnan(expert.predict(missing)[:4]).all()
    assert np.isfinite(expert.predict(missing)[4:]).all()


def test_later_labels_cannot_change_an_earlier_model():
    frame = fields()
    early = frame.race_date.lt("2023-04-01")
    model = TextExpert(min_df=5).fit(frame.loc[early])
    changed = frame.copy()
    changed.loc[~early, "won"] = 0
    other = TextExpert(min_df=5).fit(changed.loc[early])
    np.testing.assert_array_equal(model.model.weights, other.model.weights)
    assert model.model.scale == other.model.scale


def test_shared_encoding_preserves_the_fitted_model_and_keeps_training_data_out_of_artifacts():
    frame = fields()
    stored = {}
    def encode(rows, encoder):
        key = (len(rows), tuple(rows.race_id), encoder.min_df, encoder.roles, encoder.separate_fields)
        if key not in stored:
            stored[key] = encoder.fit_transform(rows), encoder
        return stored[key]
    first = TextExpert(min_df=5, roles=True, separate_fields=True).fit(frame, encoding_provider=encode)
    second = TextExpert(min_df=5, roles=True, separate_fields=True).fit(frame, encoding_provider=encode)
    direct = TextExpert(min_df=5, roles=True, separate_fields=True).fit(frame)
    assert len(stored) == 2
    np.testing.assert_array_equal(first.predict(frame), second.predict(frame))
    np.testing.assert_array_equal(first.predict(frame), direct.predict(frame))
    assert not any(hasattr(first, name) for name in ("encoding_provider", "encodings", "training_matrix"))


def test_quarter_calibration_never_warm_starts_from_later_labels(monkeypatch):
    frame = rtv_text.prepare(fields())
    warm = rtv_text.TextModel("2023Q1", np.asarray(["rank:0"]), np.ones(1), np.zeros(1), "2023-03-25", 100)
    seen = []
    def fit(x, names, won, ids, rows, initial):
        seen.append(initial)
        return np.arange(x.shape[1]), np.ones(x.shape[1]), np.zeros(x.shape[1])
    monkeypatch.setattr(rtv_text, "_fit", fit)
    list(rtv_text.fit_models(frame, ["2023Q2"], purge_days=7, warm=warm))
    assert seen[0] is None and seen[1] is warm
