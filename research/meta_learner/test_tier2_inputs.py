import pandas as pd
import pytest

from research.meta_learner import first_ensemble as fe
from research.meta_learner import screen_candidates as sc
from research.meta_learner import tier2_inputs as t
from src.modelling.spec import ModelSpec


def test_every_arm_names_registered_candidate_groups():
    assert all(group in sc.CANDIDATES for groups in t.ARMS.values() for group in groups)


def test_seed_spec_keeps_the_recipe_and_renames_repeat_seeds():
    spec = ModelSpec(id="full-063-arm", features=("a", "b"), seed=42)
    assert t.seed_spec(spec, 42) is spec
    repeat = t.seed_spec(spec, 43)
    assert (repeat.id, repeat.seed, repeat.features) == ("full-063-arm-seed-43", 43, spec.features)
    assert repeat.signature != spec.signature


def test_aligned_seeds_average_by_runner_not_by_row_order():
    first = pd.DataFrame({"race_id": ["1", "1", "2"], "horse_id": [5, 6, 7], "probability": [.2, .8, 1.]})
    second = first.iloc[[2, 1, 0]].assign(probability=[1., .6, .4])
    with pytest.raises(ValueError):
        fe.average_predictions([first, second.reset_index(drop=True)])
    averaged = fe.average_predictions(t.aligned([first, second]))
    assert averaged.probability.tolist() == pytest.approx([.3, .7, 1.])
