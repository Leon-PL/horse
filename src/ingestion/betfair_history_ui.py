"""Local account setup and historical download controls for the Data page."""
from datetime import date, timedelta
import os
from pathlib import Path
import subprocess

import config


def render_history_controls():
    import streamlit as st
    from src.ingestion.betfair_credentials import certificate_paths, connection_path, save_connection
    from src.ingestion.betfair_history import HistoryClient, HistoryError, configured, sync_history
    from src.ingestion.storage import read_json

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
                        try:
                            client = HistoryClient.certificate_login(username, password, str(cert), str(key))
                            packages = client.packages()
                            if not any(p.get("sport") == "Horse Racing" and p.get("plan") == "Basic Plan" for p in packages):
                                raise HistoryError("Required historical months are missing from My Data.")
                            save_connection(username, password, cert, key)
                            st.rerun()
                        except (HistoryError, RuntimeError, OSError):
                            st.error("Could not connect. Check the account login, uploaded certificate and access "
                                     "to Horse Racing BASIC data in Betfair's My Data page.")
        if configured():
            st.success("Historical downloads are enabled for normal source refreshes.")
            first = st.date_input("First historical race date", date.today()-timedelta(days=30))
            last = st.date_input("Last historical race date", date.today()-timedelta(days=6))
            budget = st.number_input("Maximum markets per update", 1, 10000, 500, step=100)
            if st.button("Download and import missing markets"):
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
