"""Guard: every sidebar navigation option must have a matching page branch.

A blank page bug (2026-06-13) was caused by the "🧪 Experiments" radio
option losing its emoji ("  Experiments"), so it no longer matched the
``elif page == "🧪 Experiments":`` dispatch and the page body silently
rendered nothing. This test asserts the navigation labels and the
``page ==`` dispatch literals are exactly the same set.

The UI is split into two apps — the lean serving app (``picks_app.py``)
and the training app (``training_app.py``) — so both are checked.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APPS = ["picks_app.py", "training_app.py"]


def _nav_options(text: str) -> set[str]:
    # The list literal passed to the sidebar radio: st.sidebar.radio("...", [ ... ]).
    m = re.search(r'st\.sidebar\.radio\(\s*"[^"]+",\s*\[(.*?)\]', text, re.DOTALL)
    assert m, "Could not locate the sidebar radio options list"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _dispatch_labels(text: str) -> set[str]:
    # Every `page == "..."` comparison used to dispatch a page body.
    return set(re.findall(r'page\s*==\s*"([^"]+)"', text))


@pytest.mark.parametrize("app", APPS)
def test_every_nav_option_has_a_page_branch(app):
    text = (ROOT / app).read_text(encoding="utf-8")
    nav = _nav_options(text)
    dispatch = _dispatch_labels(text)

    missing = nav - dispatch  # options the user can pick that render nothing
    assert not missing, (
        f"[{app}] Navigation options with no matching `page ==` branch (would "
        f"render a blank page): {sorted(missing)}"
    )
    orphan = dispatch - nav  # branches no option can ever reach
    assert not orphan, (
        f"[{app}] `page ==` branches not reachable from any navigation "
        f"option: {sorted(orphan)}"
    )
