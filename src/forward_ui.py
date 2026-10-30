"""Manual forward forecast page using the same recorder as the CLI."""
from datetime import timedelta
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from src.forward_store import ForwardStudy, create_study, timestamp, utcnow
from src.forward_capture import discover_events, capture_event
from src.forward_assessment import save_assessment


def render_forward_page():
    st.title("📋 Forecast record")
    st.write("Save the model's win probabilities alongside current market probabilities, "
             "then compare them when results arrive. Each attempt keeps its original inputs and times.")
    root = Path(config.BASE_DIR) / "data" / "forecasts" / "forward"
    if st.button("Start a new baseline study"):
        name = utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        try:
            with st.spinner("Pinning the current model and recording the comparison rules…"):
                study = create_study(root / name)
            st.session_state["forward_study_choice"] = str(study.path)
            st.success("Study created. Eligible recording begins on the next full UK calendar day.")
        except Exception as exc:
            st.error(f"Could not create study: {exc}")
    choices = sorted((p.parent for p in root.glob("*/protocol.sha256.json")), reverse=True)
    if not choices:
        st.info("Start a study to pin the current production model and discover upcoming races.")
        return
    choice = st.selectbox("Study", [str(p) for p in choices],
                          format_func=lambda p: Path(p).name, key="forward_study_choice")
    try:
        study = ForwardStudy(choice)
    except Exception as exc:
        st.error(f"Study registration could not be verified: {exc}")
        return
    ready = True
    try:
        study.verify()
    except Exception as exc:
        ready = False
        st.warning(f"This study cannot accept new forecasts: {exc}. Its saved attempts remain available below.")
    p = study.protocol
    st.caption(f"Model: {p['model_id']} · Eligible from {p['first_eligible_date']} · "
               "Reference: Matchbook best back prices in GBP")
    st.info("This is baseline monitoring. A formal market-parity decision needs a separately "
            "registered sample size and confirmation plan.")
    readout_path = study.path / "readout_plan.json"
    if readout_path.exists() and ready:
        from src.forward_readout import load_plan, readout_readiness
        try:
            readout_plan = load_plan(readout_path)
            progress = readout_readiness(study, readout_plan)
            pair_col, date_col = st.columns(2)
            pair_col.metric("Pilot eligible pairs", f"{progress['published_eligible_pairs']} / {readout_plan['minimum_published_pairs']}")
            date_col.metric("Pilot race dates", f"{progress['distinct_race_dates']} / {readout_plan['minimum_race_dates']}")
            st.caption(f"Pilot starts {readout_plan['eligible_start_date']}. Review follows at least "
                       f"{readout_plan['minimum_calendar_days']} calendar days and both count requirements, "
                       f"then {readout_plan['settlement_grace_days']} days for results. "
                       "This review measures coverage and the forecast gap.")
            if progress["phase"] == "ready_for_descriptive_readout":
                st.info("The registered pilot sample is ready for its descriptive review.")
        except Exception as exc:
            st.error(f"Pilot readout plan could not be verified: {exc}")
    elif readout_path.exists():
        st.caption("This retired study retains its original pilot plan and saved attempts.")
    st.write("Discover races before their recording window opens. Complete eligible recordings "
             "between 5 minutes 30 seconds and 5 minutes before the first observed start time. "
             "Other attempts remain available for monitoring.")
    if st.button("Refresh upcoming races", disabled=not ready):
        try:
            with st.spinner("Reading the current public race schedule…"):
                receipt = discover_events(study)
            st.session_state[f"forward_events:{choice}"] = receipt
        except Exception as exc:
            st.error(f"Could not refresh races: {exc}")
    receipt = st.session_state.get(f"forward_events:{choice}")
    if receipt:
        st.caption(f"Schedule checked at {receipt['received_at']}")
        if receipt["truncated"]:
            st.warning("The bounded schedule request did not cover every available event.")
        events = receipt["events"]
    else:
        events = [dict(json.loads(path.read_text())) for path in sorted((study.path / "events").glob("*.json"))]
        events = [{**event, "id": event["event_id"]} for event in events]
    events = [event for event in events if timestamp(event["first_start"]) > utcnow()]
    if events:
        by_id = {event["id"]: event for event in events}
        event_id = st.selectbox("Race", list(by_id), format_func=lambda i:
                                f"{by_id[i]['name']} · {by_id[i]['first_start']}",
                                key=f"forward_event:{choice}")
        first_start = timestamp(by_id[event_id]["first_start"])
        close = first_start-timedelta(seconds=p["horizon_seconds"])
        opening = close-timedelta(seconds=p["capture_window_seconds"])
        st.caption(f"Eligible publication window (UTC): {opening:%Y-%m-%d %H:%M:%S} to {close:%H:%M:%S}. "
                   "Start a little earlier to allow time to calculate the forecast.")
        timed = st.button("Record in eligible window", disabled=not ready or study.selected(event_id) is not None,
            help="Start within ten minutes of the recording window. This one action calculates, waits if needed, then reads the card and prices again. Keep the action running until it finishes.")
        immediate = st.button("Record fresh forecast", disabled=not ready or study.selected(event_id) is not None)
        if timed or immediate:
            try:
                message = ("Calculating now, then waiting for the recording window before refreshing the card and prices…"
                           if timed else "Fetching a fresh racecard, calculating probabilities and reading current prices…")
                with st.spinner(message):
                    result = capture_event(study, event_id, wait_for_issue_window=timed)
                st.session_state[f"forward_capture:{choice}"] = result["capture_id"]
            except Exception as exc:
                st.error(f"Capture could not finish: {exc}")
    else:
        st.caption("Refresh the schedule to find upcoming UK and Irish races.")
    records = sorted((study.path / "records").glob("*/capture.json"), reverse=True)
    if records:
        identifiers = [path.parent.name for path in records]
        selected = st.session_state.get(f"forward_capture:{choice}")
        record_id = st.selectbox("Saved attempt", identifiers,
                                index=identifiers.index(selected) if selected in identifiers else 0)
        try:
            result = study.verify_record(record_id)
            index = study.selected(result["event_id"])
            eligible = bool(index and index["capture_id"] == record_id)
            st.write(f"Published: {result['published_at']} · "
                     f"{'Selected eligible pair' if eligible else 'Monitoring attempt'}")
            reasons = result.get("reasons", []).copy()
            publication = json.loads((study.path / "records" / record_id / "publication.json").read_text())
            if publication.get("reason"):
                reasons.append(publication["reason"])
            if reasons:
                st.caption("Recording notes: " + "; ".join(reasons))
            table_path = study.path / "records" / record_id / "forecasts.parquet"
            if table_path.exists():
                frame = pd.read_parquet(table_path)
                columns = [name for name in ("horse_name", "model_probability", "market_probability", "back_odds")
                           if name in frame]
                display = frame[columns].rename(columns={"horse_name": "Horse", "model_probability": "Model",
                    "market_probability": "Market", "back_odds": "Back odds"})
                st.dataframe(display.style.format({c: "{:.1%}" for c in ("Model", "Market") if c in display}, na_rep="Unavailable"),
                             hide_index=True, width="stretch")
            st.caption(f"Saved evidence: {study.path / 'records' / record_id}")
        except Exception as exc:
            st.error(f"Saved attempt failed its integrity check: {exc}")
    st.divider()
    if st.button("Compare eligible forecasts with saved results"):
        try:
            with st.spinner("Checking complete matching fields and scoring the saved forecasts…"):
                report, output = save_assessment(study, Path(config.BASE_DIR) / "data" / "races.db")
            st.write(report["coverage"])
            if "metrics" in report:
                st.dataframe(pd.DataFrame(report["metrics"]).T, width="stretch")
            else:
                st.info("No eligible complete race pairs have settled yet.")
            st.caption(f"Assessment saved to {output}")
        except Exception as exc:
            st.error(f"Assessment could not finish: {exc}")
