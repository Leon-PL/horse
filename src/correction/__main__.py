"""Forecast correction commands.

    python -m src.correction seed --benchmark DIR [--replace]
    python -m src.correction evaluate --output DIR [--first-scored 2025-07-01] [--threads 6]
    python -m src.correction fit --as-of DATE --output DIR [--threads 6]
    python -m src.correction extend --run RUN_ID
    python -m src.correction forecast --date DATE --run RUN_ID --correction DIR [--output FILE]
    python -m src.correction status

``seed`` starts the forecast history from a benchmark's walk-forward forecasts.
``evaluate`` fits the correction on earlier quarters and scores it on later
ones, development races only. ``fit`` fits it on every settled race with a
forecast up to a date and saves it. ``extend`` adds a fitted candidate's
forecasts for races after its training cutoff to the history, so the records
keep up with results. ``forecast`` scores a day's card with the win model
and the correction. Nothing here promotes a model.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.correction import data, history
from src.correction import model as cm
from src.ingestion.storage import atomic_json, atomic_parquet

KEYS = cm.KEYS


def _inputs(args, *, development, end=None, exclude=()):
    frame, prices, tracking, receipt = data.assemble(args.store_root, development=development, end=end, exclude=exclude)
    values, columns = cm.build_inputs(frame, prices=prices, tracking=tracking)
    return frame, values, columns, receipt


def evaluate(args):
    frame, values, columns, receipt = _inputs(args, development=True)
    rows = frame.p_base.notna().to_numpy()
    scored, inputs = frame.loc[rows].reset_index(drop=True), values.loc[rows].reset_index(drop=True)
    print(f"{len(columns)} inputs, {scored.race_id.nunique()} races with a forecast from {scored.race_date.min().date()}",
          flush=True)
    adjustment, labels = cm.walk_forward(scored, inputs, columns, first_scored=args.first_scored, threads=args.threads)
    report = {"inputs": len(columns), "scored_quarters": labels, "store": receipt["store_root"],
              "fe_signature": receipt["fe_signature"], "history": history.status(),
              "result": cm.compare(scored, adjustment),
              "evaluation_status": "walk-forward on development quarters that earlier research has examined"}
    out = Path(args.output)
    kept = np.isfinite(adjustment)
    atomic_parquet(scored.loc[kept, [*KEYS, "race_date", "p_base"]].assign(adjustment=adjustment[kept]),
                   out / "adjustments.parquet")
    atomic_json(out / "report.json", report)
    print(json.dumps(report["result"], indent=2))


def fit(args):
    cutoff = pd.Timestamp(args.as_of)
    if pd.isna(cutoff) or cutoff != cutoff.normalize() or cutoff >= pd.Timestamp.now().normalize():
        raise SystemExit("--as-of must be a settled date before today")
    frame, values, columns, receipt = _inputs(args, development=False, end=args.as_of)
    model = cm.CorrectionModel(columns).fit(frame, values, threads=args.threads)
    model.lineage.update({"as_of": str(cutoff.date()), "fe_signature": receipt["fe_signature"],
                          "dataset_manifest": receipt["dataset_manifest"]["id"], "history": history.status(),
                          "performance_claim": "Fitted on all settled races; no held-out estimate for these boosters"})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format": "forecast_correction", "version": 1, "model": model}, out / "correction.joblib")
    atomic_json(out / "lineage.json", model.lineage)
    print(json.dumps({k: v for k, v in model.lineage.items() if k != "params"}, indent=2))


def extend(args):
    from src.modelling.data import load_dataset
    from src.run_store import load_run_predictor
    candidate = load_run_predictor(args.run)
    if getattr(candidate, "models", None) is None or len(candidate.models) != 1:
        raise SystemExit("The forecast history follows one win model; this run is not a single-model candidate")
    (name, member), = candidate.models.items()
    held = history.read()
    if held.empty:
        raise SystemExit("No forecast history; seed it from a benchmark first")
    cutoff = pd.Timestamp(candidate.lineage["as_of"])
    start = max(cutoff, held.race_date.max()) + pd.Timedelta(days=1)
    frame, _ = load_dataset(args.store_root, features=list(member.spec.features), development=False,
                            start=str(start.date()))
    if frame.empty:
        print(json.dumps({"added": 0, "reason": f"no settled races from {start.date()}"}))
        return
    records = frame[[*KEYS, "race_date"]].assign(probability=candidate.predict(frame))
    added = history.append(records, source=f"run:{args.run}", recipe_signature=member.spec.signature,
                           training_last_date=str(cutoff.date()))
    print(json.dumps({"added": added, **history.status()}, indent=2))


def forecast(args):
    from src.correction import serve
    from src.ingestion.pipeline import get_scraped_racecards
    from src.live_prediction import features_for_live_cards
    from src.run_store import load_run_predictor
    cards = get_scraped_racecards(date_str=args.date)
    if cards is None or cards.empty:
        raise SystemExit(f"No racecards for {args.date}")
    featured = features_for_live_cards(cards, args.date)
    model = serve.load_correction(Path(args.correction) / "correction.joblib")
    out, lineage = serve.forecast(featured, load_run_predictor(args.run), model, store_root=args.store_root)
    context = featured[[*KEYS, *[c for c in ("off_time", "track") if c in featured]]]
    out = out.merge(context.astype({k: out[k].dtype for k in KEYS}), on=KEYS, validate="one_to_one")
    target = Path(args.output or f"data/forecasts/corrected_{args.date}.parquet")
    atomic_parquet(out, target)
    atomic_json(target.with_suffix(".json"), {"date": args.date, "run": args.run, "races": int(out.race_id.nunique()),
                                              "runners": int(len(out)), **lineage})
    print(json.dumps({"file": str(target), "races": int(out.race_id.nunique()), "runners": int(len(out)), **lineage},
                     indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser("seed")
    seed.add_argument("--benchmark", required=True)
    seed.add_argument("--component", default=None)
    seed.add_argument("--replace", action="store_true", help="start a new history, discarding the present one")
    for name in ("evaluate", "fit", "extend", "forecast"):
        command = commands.add_parser(name)
        command.add_argument("--store-root", default=None)
        if name in ("evaluate", "fit"):
            command.add_argument("--output", required=True)
            command.add_argument("--threads", type=int, default=6)
    commands.choices["evaluate"].add_argument("--first-scored", default="2025-07-01")
    commands.choices["fit"].add_argument("--as-of", required=True)
    commands.choices["extend"].add_argument("--run", required=True)
    card = commands.choices["forecast"]
    card.add_argument("--date", required=True)
    card.add_argument("--run", required=True, help="the win model candidate, of the history's recipe")
    card.add_argument("--correction", required=True, help="folder written by `fit`")
    card.add_argument("--output", default=None)
    commands.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "seed":
        frame = history.seed(args.benchmark, component=args.component, replace=args.replace)
        print(json.dumps({"rows": len(frame), **history.status()}, indent=2))
    elif args.command == "status":
        print(json.dumps(history.status(), indent=2))
    else:
        {"evaluate": evaluate, "fit": fit, "extend": extend, "forecast": forecast}[args.command](args)


if __name__ == "__main__":
    main()
