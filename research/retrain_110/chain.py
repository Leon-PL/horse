"""Retrain the live win model and the Matchbook 11:00 correction on the current store (research 110).

    python -m research.retrain_110.chain [--threads 6]        # runs every step not yet done, in order
    python -m research.retrain_110.chain --step NAME          # one step

The live pair (win run ``20261114_122027_967139`` + ``data/matchbook/model``) was fitted on feature store
``ad3993eb``, before French/UAE races were swept into the ratings (2026-11-15). The store it now serves from
reads those ratings differently, so the pair is refitted with the same recipe, step for step as the original
chain in ``data/modelling/correction_20261113``:

1. ``benchmark``: walk-forward forecasts of the live recipe (same spec, native capacity) on the current store.
2. ``seed``: a new forecast history from those forecasts, in this study's own folder.
3. ``history_run`` + ``extend_history``: a model fitted to 2026-06-03 forecasts the races after it.
4. ``serving_run`` + ``extend_serving``: the new serving model, fitted to the last settled day.
5. ``correction``: the Matchbook 11:00 correction on the new history.
6. ``rehearsal_new`` / ``rehearsal_live``: research 108's development rehearsal (2025Q3-2026Q2, the registered
   rule) on the new history and on a copy of the live history, both on the current store.

Nothing here touches ``data/matchbook``, the live forecast history or ``data/runs`` entries other than the two new
runs it saves. Switching ``data/matchbook/config.json`` is a separate, manual decision.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

OUT = Path("data/research/retrain_110")
LIVE_BENCHMARK = Path("data/modelling/correction_20261113/benchmark")
HISTORY_AS_OF = "2026-06-03"
NEW_HISTORY = OUT / "forecast_history"
LIVE_COPY = OUT / "live_history_copy"
STEPS = ("benchmark", "seed", "history_run", "extend_history", "serving_run", "extend_serving", "correction",
         "rehearsal_new", "rehearsal_live")


def _status():
    path = OUT / "status.json"
    return json.loads(path.read_text()) if path.exists() else {"done": {}}


def _mark(step, **info):
    status = _status()
    status["done"][step] = {"at": datetime.now().isoformat(timespec="seconds"), **info}
    (OUT / "status.json").write_text(json.dumps(status, indent=1))


def _log(message):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    with open(OUT / "chain.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _use_history(root):
    """Point the correction's forecast history (and its settled cache) at ``root`` for this process."""
    from src.correction import history
    root = Path(root).resolve()
    history.default_root = lambda: root


def _run(args, log_name):
    with open(OUT / log_name, "a", encoding="utf-8") as log:
        code = subprocess.call([sys.executable, "-u", *args], stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise SystemExit(f"{' '.join(args[:3])} failed with exit {code}; see {OUT / log_name}")


def _last_json(log_name):
    for line in reversed((OUT / log_name).read_text(encoding="utf-8").splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No result line in {log_name}")


def benchmark(threads):
    registered = json.loads((LIVE_BENCHMARK / "registration.json").read_text())
    if len(registered["specs"]) != 1 or registered["capacity"] != "native":
        raise ValueError("The live benchmark is not the single native recipe this chain expects")
    spec = OUT / "live_spec.json"
    spec.write_text(json.dumps(registered["specs"][0], indent=1))
    _run(["-m", "src.modelling", "benchmark", "--spec", str(spec), "--capacity", "native", "--threads", str(threads),
          "--output", str(OUT / "benchmark")], "benchmark.log")
    new = json.loads((OUT / "benchmark" / "registration.json").read_text())
    if new["folds"] != registered["folds"]:
        raise ValueError("The new benchmark's folds differ from the live benchmark's")
    return {"folds": len(new["folds"])}


def seed(threads):
    from src.correction import history
    frame = history.seed(OUT / "benchmark", component="generalist", root=NEW_HISTORY, replace=True)
    return {"rows": int(len(frame)), **history.status(NEW_HISTORY)}


def _fit(as_of, name, threads, log_name):
    _run(["-m", "src.modelling", "fit", "--benchmark", str(OUT / "benchmark"), "--as-of", as_of,
          "--threads", str(threads), "--name", name], log_name)
    return _last_json(log_name)


def history_run(threads):
    return _fit(HISTORY_AS_OF, f"win 87 as of {HISTORY_AS_OF} (forecast history, research 110)", threads,
                "history_run.log")


def _extend(step):
    from src.correction.__main__ import extend
    _use_history(NEW_HISTORY)
    run = _status()["done"][step]["run_id"]
    extend(argparse.Namespace(run=run, store_root=None))
    from src.correction import history
    return {"run": run, **history.status(NEW_HISTORY)}


def extend_history(threads):
    return _extend("history_run")


def serving_run(threads):
    from src.correction import history
    last = history.status(NEW_HISTORY)["last_date"]
    return {**_fit(last, f"win 87 as of {last} (serving, research 110)", threads, "serving_run.log"), "as_of": last}


def extend_serving(threads):
    return _extend("serving_run")


def correction(threads):
    from src.correction import history
    from src.matchbook import model as mm
    _use_history(NEW_HISTORY)
    as_of = history.status(NEW_HISTORY)["last_date"]
    model = mm.fit(as_of, OUT / "matchbook_model", threads=threads)
    return {k: model.lineage.get(k) for k in ("as_of", "last_priced_race_date", "races", "fe_signature")}


def _rehearsal(root, name, threads):
    """Research 108's rehearsal, unchanged, on the forecast history at ``root``."""
    _use_history(root)
    from research.holdout_1100_107 import run as h
    from research.matchbook_variant_108.run import DROPPED
    from src.correction import model as cm
    frame, values, columns, receipt = h.build(development=True)
    columns = [c for c in columns if c not in DROPPED]
    values = values.drop(columns=list(DROPPED))
    adjustment, labels = cm.walk_forward(frame, values, columns, first_scored="2025-07-01", threads=threads)
    kept = np.isfinite(adjustment)
    frame = frame.loc[kept].reset_index(drop=True)
    p = cm.corrected(frame.p_base.to_numpy(), adjustment[kept], frame.race_id.to_numpy())
    report = {"history": str(root), "quarters": labels, "inputs": len(columns), "fe_signature": receipt["fe_signature"],
              **h.evaluate(frame, p),
              "log_loss_corrected": float(h.race_log_loss(frame, p).mean()),
              "log_loss_base": float(h.race_log_loss(frame, frame.p_base).mean())}
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=1, default=float))
    rule = report["one_tick_worse threshold 0.02"]
    return {"geo_clv": rule["geometric_mean_clv"], "median_clv": rule["median_clv"], "roi": rule["roi"]["mean"],
            "bets": rule["bets"], "log_loss_corrected": report["log_loss_corrected"],
            "log_loss_base": report["log_loss_base"]}


def rehearsal_new(threads):
    return _rehearsal(NEW_HISTORY, "rehearsal_new", threads)


def rehearsal_live(threads):
    from src.correction import history
    if not (LIVE_COPY / "history.parquet").exists():
        LIVE_COPY.mkdir(parents=True, exist_ok=True)
        for name in ("history.parquet", "meta.json"):
            shutil.copy2(history.default_root() / name, LIVE_COPY / name)
    return _rehearsal(LIVE_COPY, "rehearsal_live", threads)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--step", choices=STEPS, default=None)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step:
        started = time.monotonic()
        result = globals()[args.step](args.threads)
        _mark(args.step, seconds=round(time.monotonic() - started), **(result or {}))
        _log(f"{args.step} done: {json.dumps(result, default=str)}")
        return
    # Each step in its own process: the history redirection and loaded data never leak between steps.
    for step in STEPS:
        if step in _status()["done"]:
            continue
        _log(f"{step} started")
        code = subprocess.call([sys.executable, "-u", "-m", "research.retrain_110.chain", "--step", step,
                                "--threads", str(args.threads)])
        if code:
            _log(f"{step} FAILED (exit {code})")
            raise SystemExit(code)
    _log("chain finished")


if __name__ == "__main__":
    main()
