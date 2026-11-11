"""Do the corrections help a full-capacity generalist, or only repair the screen-capacity one?

    python -m research.error_anatomy.native_check

The corrections are learned against 400-tree screen fits. One native-capacity
full 063 forecast exists (2025Q3, 3,535 races, the benchmark's quick check).
This applies every saved correction to both forecasts on those races. A
correction that only makes up for the screen fit's missing capacity will gain
on the screen forecast and not on the native one.
"""
import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from research.meta_learner.specialists import race_losses
from src.ingestion.storage import atomic_json

NATIVE = s.STUDY / "native_quick_2025Q3/native-full-063_oof.parquet"


def main():
    native = pd.read_parquet(NATIVE, columns=[*s.KEYS, "probability"]).rename(columns={"probability": "p_native"})
    native["race_id"] = native.race_id.astype(str)
    frame = sc.base_frame()
    frame = frame.loc[frame.quarter.eq("2025Q3")].reset_index(drop=True)
    frame = s.attach(frame, native)
    frame = s.attach(frame, pd.read_parquet(OUT / "soft_corrections.parquet"))
    bounds = s.bounds_of(frame.race_id.to_numpy())
    races = frame.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    screen, full = np.log(frame.p_single.to_numpy()), np.log(frame.p_native.to_numpy())
    report = {"races": len(races), "screen_loss": float(race_losses(frame, screen).mean()),
              "native_loss": float(race_losses(frame, full).mean()), "corrections": {}}
    print(f"2025Q3, {len(races)} races: screen {report['screen_loss']:.5f}, native {report['native_loss']:.5f}")
    print(f"{'correction':<32}{'on screen':>11}{'on native':>11}   native 95% interval")
    for column in [c for c in frame if "|" in c]:
        c = frame[column].to_numpy()
        on = {name: s.paired(race_losses(frame, base + c), race_losses(frame, base), races)
              for name, base in (("screen", screen), ("native", full))}
        report["corrections"][column] = on
        n = on["native"]
        print(f"{column:<32}{on['screen']['mean']:>+11.5f}{n['mean']:>+11.5f}   "
              f"[{n['ci95'][0]:+.5f}, {n['ci95'][1]:+.5f}]")
    atomic_json(OUT / "native_check.json", report)


def stack_check(arm="soft-long-rich"):
    """The combined, weighted correction of a rebuilt stack on the screen and native 2025Q3 forecasts."""
    import json
    native = pd.read_parquet(NATIVE, columns=[*s.KEYS, "probability"]).rename(columns={"probability": "p_native"})
    native["race_id"] = native.race_id.astype(str)
    frame = sc.base_frame()
    frame = frame.loc[frame.quarter.eq("2025Q3")].reset_index(drop=True)
    frame = s.attach(frame, native)
    frame = s.attach(frame, pd.read_parquet(OUT / f"soft_stack_{arm}.parquet"))
    weights = next(row for row in json.loads((OUT / f"soft_stack_{arm}.json").read_text())["coefficients"]
                   if row["fold"] == "2025Q3")
    columns = [c for c in frame if c.endswith(":c")]
    bounds = s.bounds_of(frame.race_id.to_numpy())
    races = frame.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    bases = {"screen single": np.log(frame.p_single.to_numpy()), "screen three-seed": np.log(frame.p_average.to_numpy()),
             "native single": np.log(frame.p_native.to_numpy())}
    sets = {"all six": columns, "horse and market gap": ["horse:c", "market-gap:c"],
            "all but tracking": [c for c in columns if c != "tracking:c"]}
    report = {}
    for label, chosen in sets.items():
        total = sum(weights[c] * frame[c].to_numpy() for c in chosen)
        report[label] = {}
        for name, base in bases.items():
            r = s.paired(race_losses(frame, base + total), race_losses(frame, base), races)
            report[label][name] = r
            print(f"{label:<22} on {name:<18} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]")
    atomic_json(OUT / f"native_stack_check_{arm}.json", report)


if __name__ == "__main__":
    import sys
    stack_check() if "stack" in sys.argv else main()
