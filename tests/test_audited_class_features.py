import numpy as np
import pandas as pd

from src.audited_class_features import class_transition_features,uk_class_number


def data():
    return pd.DataFrame({"horse_id":[1,1,1,2],"race_id":[1,2,3,3],
        "race_date":pd.to_datetime(["2025-01-01","2025-01-02","2025-01-03","2025-01-03"]),
        "race_class":["5.0","Nan","3","5"],"finish_position":[1,0,1,2],
        "won":[1,0,1,0],"finish_pos_label":["","PU","",""]})


def test_unknown_and_group_class_numbers_are_not_invented_class_zero_or_two():
    actual=uk_class_number(pd.Series(["Class 1","5.0",3,None,"Nan","0","Group 2","Grade 1"]))
    np.testing.assert_array_equal(actual.iloc[:3],[1,5,3])
    assert actual.iloc[3:].isna().all()


def test_unknown_previous_start_is_not_skipped_to_recover_an_older_class():
    history=data();cards=history.iloc[[2,3]].copy()
    result=class_transition_features(cards,history)
    assert result.previous_class_known.eq(0).all()
    assert result.class_transition_known.eq(0).all()
    assert result.class_change_missing_aware.isna().all()


def test_current_unknown_class_is_not_a_false_promotion_from_class_five():
    history=data();cards=history.iloc[[1]].copy()
    result=class_transition_features(cards,history)
    assert result.previous_race_class.iloc[0]==5
    assert result.current_class_known.iloc[0]==0
    assert np.isnan(result.class_change_missing_aware.iloc[0])


def test_future_same_day_results_and_input_order_do_not_change_class_signals():
    history=data();cards=history.iloc[[2,3]].copy()
    original=class_transition_features(cards,history)
    changed=history.copy();later=changed.race_date.ge(cards.race_date.min())
    changed.loc[later,"race_class"]="1";changed.loc[later,"finish_position"]=0
    changed.loc[later,"won"]=0
    pd.testing.assert_frame_equal(original,class_transition_features(cards,changed.sample(frac=1,random_state=2)))
