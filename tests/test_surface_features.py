import numpy as np
import pandas as pd

from src.surface_features import pedigree_surface_history, synthetic_surface


def fixture():
    rows = [
        (1,"2025-01-01","A","Polytrack",1,1,""),
        (1,"2025-01-01","A","Polytrack",2,0,""),
        (2,"2025-01-02","A","Allweather",0,0,"PU"),
        (2,"2025-01-02","B","Allweather",1,1,""),
        (3,"2025-01-03","A","Turf",1,1,""),
        (3,"2025-01-03","B","Turf",2,0,""),
        (4,"2025-01-04","A","Fibresand",0,0,"NR"),
        (5,"2025-01-05","A","Tapeta",0,0,"")]
    x=pd.DataFrame(rows,columns=["race_id","race_date","sire","surface","finish_position","won","finish_pos_label"])
    x["race_date"]=pd.to_datetime(x.race_date)
    x["num_runners"]=2;x["going"]="Good";x["distance_furlongs"]=8.
    x["horse_name"]=[f"H{i}" for i in range(len(x))]
    return x


def test_actual_surface_vocabulary_is_recognised_without_confusing_beach_sand():
    actual=synthetic_surface(pd.Series(["Allweather","Polytrack","Fibresand","Tapeta","All Weather","AW","Turf","Sand",None,"?"]))
    np.testing.assert_array_equal(actual.iloc[:8],[1,1,1,1,1,1,0,0])
    assert actual.iloc[8:].isna().all()


def test_prior_counts_include_dnfs_but_not_withdrawals_or_pending_cards():
    history=fixture()
    cards=pd.DataFrame({"sire":["A","A",None,"New"], "surface":"Allweather",
                        "race_date":pd.to_datetime(["2025-01-01","2025-01-06","2025-01-06","2025-01-06"])})
    out=pedigree_surface_history(cards,history,entities=("sire",))
    assert out.sire_aw_prior_runs.iloc[0]==0
    assert out.sire_aw_prior_runs.iloc[1]==3
    assert out.sire_aw_win_rate_shrunk.iloc[1]==2/13
    assert out.sire_aw_edge.iloc[1]==2/13-4/24
    assert np.isnan(out.sire_aw_edge.iloc[2])
    assert out.sire_aw_prior_runs.iloc[3]==0


def test_card_results_future_results_same_day_and_row_order_cannot_change_inputs():
    history=fixture();cards=history.iloc[[2,3]].copy()
    original=pedigree_surface_history(cards,history,entities=("sire",))
    changed=history.copy();mask=changed.race_date.ge(cards.race_date.min())
    changed.loc[mask,"won"]=1-changed.loc[mask,"won"]
    changed.loc[mask,"finish_position"]=1
    cards["won"]=1;cards["finish_position"]=1
    pd.testing.assert_frame_equal(original,pedigree_surface_history(cards,changed.sample(frac=1,random_state=7),entities=("sire",)))
    pd.testing.assert_frame_equal(original,pedigree_surface_history(cards,history.loc[history.race_date.lt(cards.race_date.min())],entities=("sire",)))


def test_corrected_counters_match_native_stage_with_recognised_surface_labels():
    from src.feature_engineer import _add_result_history_flags, add_pedigree_features
    history=fixture()
    expected=pedigree_surface_history(history,history,entities=("sire",))
    canonical=history.copy()
    canonical.loc[synthetic_surface(canonical.surface).eq(1),"surface"]="AW"
    actual=add_pedigree_features(_add_result_history_flags(canonical))
    for name in ("sire_aw_win_rate_shrunk","sire_aw_place_rate_shrunk","sire_aw_edge"):
        np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=0,equal_nan=True)
