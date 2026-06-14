"""Guard: every sidebar navigation option must have a matching page branch.

A blank page bug (2026-06-13) was caused by the "🧪 Experiments" radio
option losing its emoji ("  Experiments"), so it no longer matched the
``elif page == "🧪 Experiments":`` dispatch and the page body silently
rendered nothing. This test asserts the navigation labels and the
``page ==`` dispatch literals are exactly the same set.
"""

import re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def _nav_options(text: str) -> set[str]:
    # The list literal passed to st.sidebar.radio("Navigation", [ ... ])
    m = re.search(r'st\.sidebar\.radio\(\s*"Navigation",\s*\[(.*?)\]', text, re.DOTALL)
    assert m, "Could not locate the Navigation radio options list in app.py"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _dispatch_labels(text: str) -> set[str]:
    # Every `page == "..."` comparison used to dispatch a page body.
    return set(re.findall(r'page\s*==\s*"([^"]+)"', text))


def test_every_nav_option_has_a_page_branch():
    text = APP.read_text(encoding="utf-8")
    nav = _nav_options(text)
    dispatch = _dispatch_labels(text)

    missing = nav - dispatch  # options the user can pick that render nothing
    assert not missing, (
        f"Navigation options with no matching `page ==` branch (would render "
        f"a blank page): {sorted(missing)}"
    )
    orphan = dispatch - nav  # branches no option can ever reach
    assert not orphan, (
        f"`page ==` branches not reachable from any navigation option: {sorted(orphan)}"
    )
