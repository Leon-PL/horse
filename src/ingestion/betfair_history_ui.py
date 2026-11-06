"""Local account setup and historical download controls for the Data page."""
from datetime import date, timedelta
import os
from pathlib import Path
import subprocess

import config


def render_history_controls():
    import streamlit as st
    from src.ingestion.betfair_credentials import certificate_paths, connection_path
    from src.ingestion.betfair_history import HistoryError, configured, connect_account, sync_history
    from src.ingestion.storage import read_json, writer_lock

    with st.expander("Historical exchange data"):
        st.write("Automatically fetch full historical market files, including timed prices, "
                 "for British and Irish win and place markets.")
        st.caption("Uses your existing BASIC historical-data access. A paid Live App Key is not required. "
                   "Historical files are normally published five days after event settlement.")
        cert, key = certificate_paths()
        if not cert.exists() or not key.exists():
            st.write("First, create a login certificate for this computer.")
            if st.button("Create login certificate"):
                try:
                    command = ["pwsh", "-NoProfile", "-NonInteractive", "-File",
                               str(Path(config.BASE_DIR)/"scripts/setup_betfair_history.ps1"), "-CreateCertificate"]
                    result = subprocess.run(command, capture_output=True, timeout=60,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    if result.returncode:
                        st.error("Certificate creation did not finish. See the historical-data setup guide.")
                    else:
                        st.rerun()
                except (OSError, subprocess.TimeoutExpired):
                    st.error("Certificate creation needs PowerShell 7 on this computer.")
        else:
            st.download_button("Save public certificate to upload", cert.read_bytes(),
                               file_name="betfair-history-client.crt", mime="application/x-pem-file")
            st.markdown("Upload that public certificate once in [Betfair account security]"
                "(https://myaccount.betfair.com/accountdetails/mysecurity?showAPI=1), "
                "under **Automated Betting Program Access**. Then connect below.")
            if not connection_path().exists():
                result = read_json(connection_path().with_name("connection_status.json"))
                with st.form("betfair_history_connection", clear_on_submit=True):
                    username = st.text_input("Betfair username")
                    password = st.text_input("Betfair password", type="password")
                    st.caption("Login details are saved encrypted for this Windows account. "
                               "Future source refreshes can sign in automatically.")
                    submitted = st.form_submit_button("Connect for automatic downloads")
                if submitted:
                    if not username or not password:
                        st.error("Enter your Betfair username and password.")
                    else:
                        result = connect_account(username, password, cert, key)
                        if result["state"] == "connected":
                            st.rerun()
                if result and result.get("state") == "failed":
                    labels = {"certificate_login": "Betfair sign-in failed",
                              "historical_access": "Historical data access failed",
                              "save_connection": "Saving the connection failed"}
                    st.error(f"{labels.get(result.get('stage'), 'Connection failed')}: {result['message']}")
                    if result.get("stage") == "historical_access":
                        st.link_button("Open Betfair My Data", "https://historicdata.betfair.com/#/mydata")
        if configured():
            st.success("Historical downloads are enabled for normal source refreshes.")
            root = Path(config.DATA_DIR)/"betfair/history"
            with writer_lock(root/".backfill.lock") as available:
                backfill_running = not available
            if backfill_running:
                backfill = read_json(root/"backfill_status.json")
                if backfill.get("replaying_archives"):
                    st.info("Historical catch-up is reusing existing downloads while they are consolidated.")
                else:
                    counts = read_json(root/"status.json").get("files", backfill.get("files", {}))
                    st.info(f"Historical catch-up is running: {counts.get('published', 0):,} markets imported, "
                            f"{counts.get('pending', 0):,} queued.")
            first = st.date_input("First historical race date", date.today()-timedelta(days=30))
            last = st.date_input("Last historical race date", date.today()-timedelta(days=6))
            budget = st.number_input("Maximum markets per update", 1, 10000, 500, step=100)
            if st.button("Download and import missing markets", disabled=backfill_running):
                try:
                    with st.spinner("Downloading markets and updating price history…"):
                        report = sync_history(first, last, budget=int(budget))
                    if report["state"] == "ok":
                        st.success("Historical price update finished.")
                    else:
                        st.warning("Progress saved; see the remaining work below.")
                    st.json(report, expanded=False)
                except (HistoryError, ValueError) as error:
                    st.error(str(error))
            status = read_json(Path(config.DATA_DIR)/"betfair/history/status.json")
            if status:
                st.caption(f"Last update: {status.get('checked_at', '?')} · {status.get('state', '?')}")
            if connection_path().exists() and st.button("Disconnect historical downloads"):
                connection_path().unlink()
                st.rerun()
