"""Headless smoke test: boot the app and switch through key pages."""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest

t0 = time.perf_counter()
at = AppTest.from_file("app.py", default_timeout=600)
at.run()
print(f"initial run: {time.perf_counter() - t0:.1f}s, exception={bool(at.exception)}")
if at.exception:
    print(at.exception[0].value)
    raise SystemExit(1)

pages = [p for p in at.sidebar.radio[0].options]
for page in pages:
    t1 = time.perf_counter()
    at.sidebar.radio[0].set_value(page).run()
    status = "EXC" if at.exception else "ok"
    print(f"{page:28s} rerun={time.perf_counter() - t1:5.1f}s {status}")
    if at.exception:
        print(at.exception[0].value)
print("done")
