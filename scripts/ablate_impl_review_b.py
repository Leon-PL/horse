"""Round 2 of the repair ablation; same screen, folds and baseline as round 1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate_impl_review as base  # noqa: E402

base.CASE = base.ROOT / "data/audit/feature_impl_review_047b"
SHRUNK = ["jockey_going_wr", "jockey_dist_cat_wr", "trainer_going_wr", "trainer_going_wr_vs_field",
          "trainer_going_wr_vs_best_in_race", "trainer_class_wr_vs_field", "trainer_class_wr_z_in_race",
          "trainer_class_wr_vs_best_in_race", "trainer_dist_cat_wr_vs_best_in_race", "trainer_track_win_rate_vs_field",
          "trainer_track_win_rate_z_in_race", "trainer_track_win_rate_vs_best_in_race",
          "jockey_track_win_rate_vs_best_in_race", "connections_relative_score"]
base.CANDIDATES = {
    "shrunk_dims": {c: c + "__fix" for c in SHRUNK},
    "or_observed": {c: c + "__fix" for c in ["or_rank_in_race", "or_vs_second_top", "field_std_or"]},
    "rank_share": {c: c + "__fix" for c in ["horse_glicko_rank", "horse_elo_rank"]},
    # Exact duplicates for a tree (monotone transforms or a re-centred centred column).
    "dedupe": {c: None for c in ["base_win_rate", "days_log", "horse_margin_elo_momentum_vs_field_vs_field",
                                 "trainer_elo_vs_field_vs_field", "horse_elo_momentum_vs_field_vs_best_in_race"]},
}

if __name__ == "__main__":
    base.main()
